#!/usr/bin/env python3
"""Nearest-neighbour baseline scored on the ABSOLUTE (global-scale) axis.

Zero-argument runnable::

    python evaluation/eval_nntest.py

Why this script exists
----------------------
Every existing eval in this repository scores a predictor against a target that
has been max-normalized *per location* (``utils.loss.normalize_mag_map``), so
the reported NMSE is a PATTERN error: it is blind to how much total power a map
carries.  Under that metric the trivial nearest-train-neighbour predictor is
very strong (-23.81 dB mean on ``asu_campus_16by64_lt``), because a neighbouring
location has essentially the same beam pattern even when it has quite a
different received level.

Before deciding whether to retrain with an absolute-scale loss, we need to know
what that same trivial baseline scores when NOTHING is renormalized -- when the
prediction and the ground truth are both left in the dataset's single global
scale.  That is the metric a retrained model would be judged on, so the number
here is the bar an absolute-scale model has to clear, and the gap between the
two metrics says how much of the "NN is strong" story was pattern-only.

What is computed
----------------
Data only.  No model is loaded, nothing is rendered, nothing is trained.

For every scored test location, with the FULL train set as the reference and
3-D Euclidean distance in ORIGINAL meters (the same neighbour rule as
``eval_t1``/``eval_density.nearest_neighbour_maps``, which also uses
``scipy.spatial.cKDTree`` on raw coordinates):

* ``1-NN``  -- the nearest train location's map, VERBATIM.  No rescaling.
* ``2-NN``  -- the linear-domain mean of the two nearest train maps.  On the
               measured data this was the strongest trivial baseline, so it is
               carried along as a second row.

and two metrics side by side:

* absolute NMSE : ||pred - X||^2 / ||X||^2                 (level + pattern)
* shape NMSE    : ||N(pred) - N(X)||^2 / ||N(X)||^2        (pattern only)

with ``N(.) = normalize_mag_map(., eps=1e-8)`` imported from ``utils.loss`` --
the shape row is therefore the *same* arithmetic ``eval_baseline_rt.score_prediction``
runs, and it is checked against the published ``eval_t1`` number before any
absolute number is printed.

Two supporting readouts separate the two metrics' difference into its parts:

* level ratio ``10*log10(max(pred)/max(X))`` per location -- pure level
  mismatch, with the pattern quotiented out.  If the absolute-vs-shape gap is
  a level story, this is large; if it is pattern drift under a shared level,
  this is small.
* absolute NMSE binned by nearest-train distance -- whether the absolute error
  is a property of sparse test locations or of the whole grid.

Outputs
-------
``analysis/eval_nntest/<dataset_name>/summary.csv``      one row per predictor
``analysis/eval_nntest/<dataset_name>/per_location.csv`` one row per scored location
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Path setup -- the eval scripts live in evaluation/ in some checkouts and at
# the repo root in others; anchor on the directory that actually holds dataset/.
# ---------------------------------------------------------------------------
def _find_repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = here
    for _ in range(4):
        if os.path.isdir(os.path.join(candidate, "dataset")):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    return here


REPO_ROOT = _find_repo_root()
for _entry in (REPO_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


# ---------------------------------------------------------------------------
# Fixed inputs
# ---------------------------------------------------------------------------
DATASETS: Tuple[str, ...] = (
    # The sanity-gated dataset comes first on purpose: its shape NMSE is the
    # published eval_t1 number, so a metric bug is caught before anything else
    # is reported.
    "dataset/asu_campus_16by64_lt",
    "dataset/asu_campus_16by64_lt_stride2",
)

OUTPUT_ROOT = os.path.join(REPO_ROOT, "analysis", "eval_nntest")

# ``utils.loss.normalize_mag_map`` default, and ``eval_render.EPS``: the floor a
# map's own maximum is clamped to, and the threshold below which a map counts as
# zero-power and is dropped from scoring (a zero map makes the NMSE denominator
# degenerate).
EPS = 1e-8
# ``eval_baseline_rt.score_prediction`` clamps the NMSE ratio here before the
# log so a numerically exact prediction cannot produce -inf dB.
RATIO_FLOOR = 1e-12

# The NMSE denominator's floor.  ``score_prediction`` clamps it at EPS, which is
# safe there ONLY because its target is max-normalized and therefore always
# carries energy >= 1.  On the absolute axis the target keeps the dataset's full
# dynamic range: 70 of the 3947 asu_campus_16by64_lt test maps have raw energy
# below 1e-8 (minimum 2.8e-11), and an EPS floor silently replaces their true
# denominator, reporting one of them at -25.2 dB when its absolute NMSE is
# actually +0.27 dB.  So the absolute metric floors at the smallest positive
# double instead -- a guard against exact zero and nothing else -- while the
# shape metric keeps the EPS floor so it stays bit-identical to the published
# eval_t1 arithmetic.
ABS_DENOMINATOR_FLOOR = float(np.finfo(np.float64).tiny)

# Sanity gate: the 1-NN shape NMSE on the original dataset has to reproduce the
# published T1 table.  Read from the T1 output when it is present so the two
# cannot silently drift apart; the literal is that file's value as of 2026-08-31.
SANITY_DATASET = "dataset/asu_campus_16by64_lt"
SANITY_ROW = "Nearest neighbor"
SANITY_COLUMN = "nmse_mean_dB"
SANITY_TABLE_CSV = os.path.join(REPO_ROOT, "analysis", "eval_t1", "t1_table.csv")
SANITY_FALLBACK_DB = -23.811932
SANITY_TOLERANCE_DB = 0.1

# Nearest-train-distance bins for the absolute-NMSE breakdown, in meters.
DIST_BIN_EDGES: Tuple[float, ...] = (0.0, 0.75, 1.25, 2.0, 3.0, float("inf"))
# Positions are stored as float32, so a grid whose nominal spacing lands exactly
# on a bin edge computes as 1.99996 .. 2.0000001 rather than 2.0 -- on the
# stride-2 grid that split ONE physical distance across two bins (101 vs 890
# locations).  Distances within this tolerance of an edge are snapped onto it
# before binning; the tolerance is far below any real difference in spacing and
# far above the float32 noise.
BIN_EDGE_TOL_M = 1e-3

ROW_1NN = "1-NN (verbatim)"
ROW_2NN = "2-NN mean"

SUMMARY_COLUMNS: Tuple[str, ...] = (
    "dataset",
    "predictor",
    "n_test",
    "n_scored",
    "n_skipped_zero_power",
    "abs_nmse_mean_dB",
    "abs_nmse_median_dB",
    "abs_nmse_p5_dB",
    "abs_nmse_p95_dB",
    "shape_nmse_mean_dB",
    "shape_nmse_median_dB",
    "shape_nmse_p5_dB",
    "shape_nmse_p95_dB",
    "abs_minus_shape_mean_dB",
    "level_ratio_absmean_dB",
    "level_ratio_abs_p95_dB",
    "level_ratio_signed_mean_dB",
    "level_ratio_signed_median_dB",
    "level_only_nmse_mean_dB",
    "level_only_nmse_median_dB",
    "level_term_larger_fraction",
    "nn_distance_mean_m",
    "nn_distance_median_m",
    "nn_distance_max_m",
)

PER_LOCATION_COLUMNS: Tuple[str, ...] = (
    "test_index",
    "x",
    "y",
    "z",
    "nn_distance",
    "abs_nmse_1nn",
    "shape_nmse_1nn",
    "abs_nmse_2nn",
    "shape_nmse_2nn",
    "level_ratio_db_1nn",
)


# ---------------------------------------------------------------------------
# Per-map max normalization -- imported, not reimplemented, so the shape row is
# provably the same N(.) every other eval uses.
# ---------------------------------------------------------------------------
try:
    import torch  # noqa: E402
    from utils.loss import normalize_mag_map as _normalize_mag_map  # noqa: E402

    NORMALIZER_SOURCE = "utils.loss.normalize_mag_map (imported)"

    def normalize_maps(maps: np.ndarray) -> np.ndarray:
        """``N(A) = A / max(amax(A), EPS)`` per map, via the shipped helper."""
        tensor = torch.from_numpy(np.ascontiguousarray(maps))
        return _normalize_mag_map(tensor, eps=EPS).numpy()

except ImportError:  # pragma: no cover - torch-free fallback
    NORMALIZER_SOURCE = "numpy replica of utils.loss.normalize_mag_map (torch unavailable)"

    def normalize_maps(maps: np.ndarray) -> np.ndarray:
        scale = np.maximum(
            maps.reshape(maps.shape[0], -1).max(axis=1), EPS
        ).reshape((-1,) + (1,) * (maps.ndim - 1))
        return maps / scale


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def to_db(ratio: np.ndarray) -> np.ndarray:
    """Linear ratio -> dB, on the same clamped log as ``score_prediction``."""
    return 10.0 * np.log10(np.maximum(np.asarray(ratio, dtype=np.float64), RATIO_FLOOR))


def nmse_db(
    prediction: np.ndarray, target: np.ndarray, denominator_floor: float
) -> np.ndarray:
    """Per-location ``||pred - target||^2 / ||target||^2`` in dB.

    ``denominator_floor`` is EPS for the shape metric (parity with
    ``score_prediction``) and ``ABS_DENOMINATOR_FLOOR`` for the absolute one --
    see that constant for why the two cannot share a floor.
    """
    count = prediction.shape[0]
    pred_flat = prediction.reshape(count, -1)
    target_flat = target.reshape(count, -1)
    numerator = np.sum((pred_flat - target_flat) ** 2, axis=1)
    denominator = np.maximum(np.sum(target_flat ** 2, axis=1), denominator_floor)
    return to_db(numerator / denominator)


def map_peak(maps: np.ndarray) -> np.ndarray:
    return maps.reshape(maps.shape[0], -1).max(axis=1)


def distribution(values: np.ndarray) -> Dict[str, float]:
    """mean / median / p5 / p95 of an already-in-dB per-location array."""
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p5": float(np.percentile(values, 5.0)),
        "p95": float(np.percentile(values, 95.0)),
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_split(dataset_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """``(positions (N,3) meters float64, magnitude (N,Nr,Nt) float64)``."""
    contents = sio.loadmat(os.path.join(dataset_dir, f"{split}.mat"))
    positions = np.asarray(contents["positions"], dtype=np.float64)
    magnitude = np.asarray(contents["magnitude"], dtype=np.float64)
    if positions.shape[0] != magnitude.shape[0]:
        raise AssertionError(
            f"[eval_nntest] {dataset_dir}/{split}.mat: {positions.shape[0]} positions "
            f"but {magnitude.shape[0]} maps."
        )
    return positions, magnitude


def read_sanity_reference() -> Tuple[float, str]:
    """The published eval_t1 1-NN shape NMSE, from its CSV when available."""
    if os.path.exists(SANITY_TABLE_CSV):
        with open(SANITY_TABLE_CSV, newline="") as handle:
            for record in csv.DictReader(handle):
                if record.get("method") == SANITY_ROW:
                    value = record.get(SANITY_COLUMN)
                    if value not in (None, ""):
                        return float(value), os.path.relpath(SANITY_TABLE_CSV, REPO_ROOT)
    return SANITY_FALLBACK_DB, "hard-coded eval_t1 value"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_dataset(dataset_dir: str) -> Dict[str, object]:
    """Score both NN rows on one dataset.  Pure data; nothing is printed here."""
    train_positions, train_magnitude = load_split(dataset_dir, "train")
    test_positions, test_magnitude = load_split(dataset_dir, "test")

    if train_magnitude.shape[1:] != test_magnitude.shape[1:]:
        raise AssertionError(
            f"[eval_nntest] {dataset_dir}: train maps are {train_magnitude.shape[1:]} "
            f"but test maps are {test_magnitude.shape[1:]}."
        )
    if train_positions.shape[0] < 2:
        raise AssertionError(f"[eval_nntest] {dataset_dir}: need >= 2 train locations.")

    # Zero-power ground truth makes the NMSE denominator degenerate; drop those
    # locations and report how many, exactly as TestGroundTruth does.
    peak_target = map_peak(test_magnitude)
    valid_indices = np.nonzero(peak_target > EPS)[0].astype(np.int64)
    num_skipped = int(test_magnitude.shape[0] - valid_indices.size)

    target = test_magnitude[valid_indices]
    positions = test_positions[valid_indices]

    # Neighbour rule: 3-D Euclidean in ORIGINAL meters, full train set --
    # ``eval_density.nearest_neighbour_indices``, which is a plain
    # ``cKDTree(train).query(test, k=...)`` on raw coordinates.
    #
    # The k=1 query is NOT redundant with the first column of the k=2 query.
    # On a regular grid a test point is routinely equidistant from two train
    # points (3548 of the 3947 locations of asu_campus_16by64_lt are), and
    # cKDTree breaks those ties differently depending on k -- picking the k=2
    # first column instead moves the 1-NN shape NMSE by 0.23 dB and the row
    # stops reproducing eval_t1.  So the 1-NN row is taken from its own k=1
    # query, exactly the call eval_t1 makes.
    tree = cKDTree(train_positions)
    nn_distance, first_index = tree.query(positions, k=1)
    nn_distance = np.asarray(nn_distance, dtype=np.float64).reshape(-1)
    first_index = np.asarray(first_index, dtype=np.int64).reshape(-1)

    # Second neighbour: the closest train point OTHER than the one the 1-NN row
    # picked, so the 2-NN row is provably "the 1-NN prediction plus one more
    # map" rather than a differently tie-broken pair.
    k_wide = min(3, int(train_positions.shape[0]))
    _, wide_index = tree.query(positions, k=k_wide)
    wide_index = np.asarray(wide_index, dtype=np.int64).reshape(-1, k_wide)
    second_index = wide_index[:, 0].copy()
    same_as_first = second_index == first_index
    for column in range(1, k_wide):
        if not same_as_first.any():
            break
        replacement = wide_index[:, column]
        second_index = np.where(same_as_first, replacement, second_index)
        same_as_first = second_index == first_index
    if bool((second_index == first_index).any()):
        raise AssertionError(
            f"[eval_nntest] {dataset_dir}: could not find a distinct second "
            "nearest train location for every test point."
        )

    first = train_magnitude[first_index]
    second = train_magnitude[second_index]

    predictions: Dict[str, np.ndarray] = {
        # Verbatim: the global scale of the train map is carried through
        # untouched, which is the whole point of the absolute axis.
        ROW_1NN: first,
        # Linear-domain average of the two nearest maps.
        ROW_2NN: 0.5 * (first + second),
    }

    target_normalized = normalize_maps(target)
    peak_scored = map_peak(target)
    # Visible evidence for the ABS_DENOMINATOR_FLOOR choice on this dataset.
    low_energy_targets = int(
        np.sum(np.sum(target.reshape(target.shape[0], -1) ** 2, axis=1) < EPS)
    )

    rows: Dict[str, Dict[str, object]] = {}
    for name, prediction in predictions.items():
        abs_db = nmse_db(prediction, target, ABS_DENOMINATOR_FLOOR)
        shape_db = nmse_db(normalize_maps(prediction), target_normalized, EPS)
        level_db = to_db(np.maximum(map_peak(prediction), EPS) / np.maximum(peak_scored, EPS))
        # Level/pattern split.  Writing P = c*N(P) and X = m*N(X), the absolute
        # NMSE is exactly ||r*N(P) - N(X)||^2 / ||N(X)||^2 with r = c/m, and the
        # shape NMSE is that same expression at r = 1.  The level ratio is
        # therefore the ONLY thing separating the two metrics.  This row asks
        # how much of the absolute error the level alone can account for: the
        # NMSE a prediction with a PERFECT pattern but this location's level
        # error would score, which is simply (r - 1)^2.
        level_only_db = to_db((10.0 ** (level_db / 10.0) - 1.0) ** 2)
        rows[name] = {
            "abs_db": abs_db,
            "shape_db": shape_db,
            "level_db": level_db,
            "level_only_db": level_only_db,
            "abs": distribution(abs_db),
            "shape": distribution(shape_db),
            "level_only": distribution(level_only_db),
        }

    return {
        "dataset": dataset_dir,
        "n_test": int(test_magnitude.shape[0]),
        "n_scored": int(valid_indices.size),
        "n_skipped_zero_power": num_skipped,
        "n_train": int(train_positions.shape[0]),
        "low_energy_targets": low_energy_targets,
        "beam_shape": tuple(int(v) for v in test_magnitude.shape[1:]),
        "valid_indices": valid_indices,
        "positions": positions,
        "nn_distance": nn_distance,
        "rows": rows,
    }


def snap_to_bin_edges(nn_distance: np.ndarray) -> Tuple[np.ndarray, int]:
    """Pull distances that sit a float32 hair off a bin edge onto that edge."""
    snapped = np.asarray(nn_distance, dtype=np.float64).copy()
    moved = 0
    for edge in DIST_BIN_EDGES:
        if not np.isfinite(edge):
            continue
        close = np.abs(snapped - edge) < BIN_EDGE_TOL_M
        moved += int(np.sum(close & (snapped != edge)))
        snapped[close] = edge
    return snapped, moved


def distance_bin_table(result: Dict[str, object]) -> List[Dict[str, object]]:
    """Absolute NMSE per nearest-train-distance bin."""
    nn_distance, _ = snap_to_bin_edges(result["nn_distance"])
    rows = result["rows"]
    table: List[Dict[str, object]] = []
    for low, high in zip(DIST_BIN_EDGES[:-1], DIST_BIN_EDGES[1:]):
        mask = (nn_distance >= low) & (nn_distance < high)
        label = f"{low:.2f}-{high:.2f}" if np.isfinite(high) else f"{low:.2f}+"
        entry: Dict[str, object] = {
            "label": label,
            "count": int(mask.sum()),
            "mean_distance_m": float(np.mean(nn_distance[mask])) if mask.any() else float("nan"),
        }
        for name, row in rows.items():
            entry[name] = float(np.mean(row["abs_db"][mask])) if mask.any() else float("nan")
        table.append(entry)
    return table


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_dataset_block(result: Dict[str, object]) -> None:
    dataset = str(result["dataset"])
    rows = result["rows"]
    nn_distance = result["nn_distance"]

    print("=" * 96)
    print(f"[eval_nntest] {dataset}")
    print("=" * 96)
    print(
        f"  test locations {result['n_test']} "
        f"(scored {result['n_scored']}, skipped zero-power {result['n_skipped_zero_power']})"
        f" | train locations {result['n_train']} (full set)"
        f" | beam grid {result['beam_shape'][0]} x {result['beam_shape'][1]}"
    )
    if result["low_energy_targets"]:
        print(
            f"  {result['low_energy_targets']} scored target maps carry raw energy below "
            f"{EPS:g}; the absolute NMSE floors its denominator at "
            f"{ABS_DENOMINATOR_FLOOR:.3g}, not EPS, so they are scored on their true energy."
        )
    print("")

    header = (
        f"  {'predictor':<16} {'metric':<9} "
        f"{'mean':>9} {'median':>9} {'p5':>9} {'p95':>9}   [dB]"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in (ROW_1NN, ROW_2NN):
        row = rows[name]
        for metric_label, key in (("absolute", "abs"), ("shape", "shape")):
            stats = row[key]
            print(
                f"  {name if metric_label == 'absolute' else '':<16} {metric_label:<9} "
                f"{stats['mean']:9.3f} {stats['median']:9.3f} "
                f"{stats['p5']:9.3f} {stats['p95']:9.3f}"
            )
        print(
            f"  {'':<16} {'gap':<9} "
            f"{row['abs']['mean'] - row['shape']['mean']:9.3f} "
            f"{row['abs']['median'] - row['shape']['median']:9.3f}"
            "                        (absolute - shape)"
        )
    print("")

    print(
        f"  nearest-train distance [m] : mean {np.mean(nn_distance):.4f}"
        f" | median {np.median(nn_distance):.4f}"
        f" | max {np.max(nn_distance):.4f}"
    )
    for name in (ROW_1NN, ROW_2NN):
        level = rows[name]["level_db"]
        print(
            f"  level ratio {name:<16} : mean|.| {np.mean(np.abs(level)):8.3f} dB"
            f" | p95|.| {np.percentile(np.abs(level), 95.0):8.3f} dB"
            f" | signed mean {np.mean(level):8.3f} dB"
            f" | signed median {np.median(level):8.3f} dB"
        )
    print("")

    print("  level vs. pattern  (absolute NMSE = shape NMSE at level ratio r = 1)")
    print(f"    {'predictor':<18} {'absolute':>10} {'shape':>10} {'level-only':>12}   [dB, mean]")
    for name in (ROW_1NN, ROW_2NN):
        row = rows[name]
        share = float(np.mean(row["level_only_db"] > row["shape_db"]))
        print(
            f"    {name:<18} {row['abs']['mean']:>10.3f} {row['shape']['mean']:>10.3f} "
            f"{row['level_only']['mean']:>12.3f}   "
            f"(level term is the larger one at {100.0 * share:.1f}% of locations)"
        )
    print("")

    _, snapped_count = snap_to_bin_edges(nn_distance)
    print("  absolute NMSE by nearest-train distance [dB, mean]")
    if snapped_count:
        print(
            f"    (note: {snapped_count} distances sat within {BIN_EDGE_TOL_M:g} m of a bin "
            "edge -- float32 position storage -- and were snapped onto it before binning)"
        )
    print(
        f"    {'bin [m]':<12} {'count':>7} {'mean d [m]':>11} "
        f"{ROW_1NN:>17} {ROW_2NN:>17}"
    )
    for entry in distance_bin_table(result):
        if entry["count"] == 0:
            print(f"    {entry['label']:<12} {entry['count']:>7}          --                --                --")
            continue
        print(
            f"    {entry['label']:<12} {entry['count']:>7} "
            f"{entry['mean_distance_m']:>11.4f} "
            f"{entry[ROW_1NN]:>17.3f} {entry[ROW_2NN]:>17.3f}"
        )
    print("")


def write_outputs(result: Dict[str, object]) -> Tuple[str, str]:
    dataset = str(result["dataset"])
    output_dir = os.path.join(OUTPUT_ROOT, os.path.basename(os.path.normpath(dataset)))
    os.makedirs(output_dir, exist_ok=True)

    rows = result["rows"]
    nn_distance = result["nn_distance"]

    summary_path = os.path.join(output_dir, "summary.csv")
    with open(summary_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUMMARY_COLUMNS)
        for name in (ROW_1NN, ROW_2NN):
            row = rows[name]
            level = row["level_db"]
            writer.writerow(
                [
                    dataset,
                    name,
                    result["n_test"],
                    result["n_scored"],
                    result["n_skipped_zero_power"],
                    f"{row['abs']['mean']:.6f}",
                    f"{row['abs']['median']:.6f}",
                    f"{row['abs']['p5']:.6f}",
                    f"{row['abs']['p95']:.6f}",
                    f"{row['shape']['mean']:.6f}",
                    f"{row['shape']['median']:.6f}",
                    f"{row['shape']['p5']:.6f}",
                    f"{row['shape']['p95']:.6f}",
                    f"{row['abs']['mean'] - row['shape']['mean']:.6f}",
                    f"{float(np.mean(np.abs(level))):.6f}",
                    f"{float(np.percentile(np.abs(level), 95.0)):.6f}",
                    f"{float(np.mean(level)):.6f}",
                    f"{float(np.median(level)):.6f}",
                    f"{row['level_only']['mean']:.6f}",
                    f"{row['level_only']['median']:.6f}",
                    f"{float(np.mean(row['level_only_db'] > row['shape_db'])):.6f}",
                    f"{float(np.mean(nn_distance)):.6f}",
                    f"{float(np.median(nn_distance)):.6f}",
                    f"{float(np.max(nn_distance)):.6f}",
                ]
            )

    per_location_path = os.path.join(output_dir, "per_location.csv")
    positions = result["positions"]
    valid_indices = result["valid_indices"]
    one = rows[ROW_1NN]
    two = rows[ROW_2NN]
    with open(per_location_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PER_LOCATION_COLUMNS)
        for rank in range(int(valid_indices.size)):
            writer.writerow(
                [
                    int(valid_indices[rank]),
                    f"{positions[rank, 0]:.6f}",
                    f"{positions[rank, 1]:.6f}",
                    f"{positions[rank, 2]:.6f}",
                    f"{nn_distance[rank]:.6f}",
                    f"{one['abs_db'][rank]:.6f}",
                    f"{one['shape_db'][rank]:.6f}",
                    f"{two['abs_db'][rank]:.6f}",
                    f"{two['shape_db'][rank]:.6f}",
                    f"{one['level_db'][rank]:.6f}",
                ]
            )

    return summary_path, per_location_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"[eval_nntest] repo root      : {REPO_ROOT}")
    print(f"[eval_nntest] normalizer     : {NORMALIZER_SOURCE}")
    print(f"[eval_nntest] EPS            : {EPS:g}  (max-clamp floor and zero-power threshold)")
    print("")

    present: List[str] = []
    for dataset in DATASETS:
        dataset_dir = os.path.join(REPO_ROOT, dataset)
        missing = [
            split
            for split in ("train", "test")
            if not os.path.exists(os.path.join(dataset_dir, f"{split}.mat"))
        ]
        if missing:
            print(f"[eval_nntest] SKIP {dataset}: missing {', '.join(m + '.mat' for m in missing)}")
            continue
        present.append(dataset)

    if not present:
        print("[eval_nntest] No dataset from DATASETS is present; nothing to do.")
        return 1

    results: List[Dict[str, object]] = []
    for dataset in present:
        result = evaluate_dataset(os.path.join(REPO_ROOT, dataset))
        result["dataset"] = dataset
        results.append(result)

    # -- sanity gate: the shape row has to reproduce the published eval_t1
    #    number BEFORE any absolute number is shown --------------------------
    reference, reference_source = read_sanity_reference()
    gated = [r for r in results if str(r["dataset"]) == SANITY_DATASET]
    print("-" * 96)
    print("[eval_nntest] SANITY  1-NN shape NMSE vs. eval_t1")
    print("-" * 96)
    if not gated:
        print(
            f"  {SANITY_DATASET} is not present, so the shape metric could not be "
            "checked against eval_t1.  Absolute numbers below are UNGATED."
        )
    else:
        measured = float(gated[0]["rows"][ROW_1NN]["shape"]["mean"])
        delta = measured - reference
        print(f"  reference ({reference_source}): {reference:9.4f} dB")
        print(f"  measured  (this script)       : {measured:9.4f} dB")
        print(f"  delta                         : {delta:+9.4f} dB "
              f"(tolerance +/- {SANITY_TOLERANCE_DB:g} dB)")
        if abs(delta) > SANITY_TOLERANCE_DB:
            print("")
            print("  FAIL: the shape NMSE does not reproduce eval_t1.  The metric or the")
            print("        neighbour rule disagrees with the published table, so the")
            print("        absolute numbers would not be trustworthy.  Stopping here;")
            print("        no absolute results printed and no files written.")
            return 1
        print("  PASS")
    print("")

    for result in results:
        print_dataset_block(result)
        summary_path, per_location_path = write_outputs(result)
        print(f"  wrote {os.path.relpath(summary_path, REPO_ROOT)}")
        print(f"  wrote {os.path.relpath(per_location_path, REPO_ROOT)}")
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
