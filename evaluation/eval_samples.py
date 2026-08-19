#!/usr/bin/env python3
"""D5 -- DeepMIMO qualitative rendering samples.

Draws the paper's DeepMIMO rendering-sample figure from the 100 %-density
MIMO-GS checkpoint ``outputs/density/mimogs/model_100.pth``, scored on the
ORIGINAL full test set of ``dataset/asu_campus_16by64_lt``.

Zero-argument runnable::

    python eval_samples.py                    # candidate listing + gallery
    python eval_samples.py --spots 12,34,56   # final publication figure

Nothing in the repository is modified.  Every number comes from
``evaluation/eval_render.py`` -- directly, or through
``evaluation/eval_baseline_rt.score_prediction`` and the shared plumbing of
``evaluation/eval_density.py`` (``TestGroundTruth``, ``load_mimogs`` /
``load_mlp``, the render paths, the figure conventions), so the per-location
shape NMSE printed here is produced by exactly the same arithmetic as T1, D1
and D3.

Candidate selection
-------------------
A qualitative figure should show TYPICAL rendering behaviour, not the tails, so
the candidate pool is the quality band between the 25th and the 75th percentile
of the per-location MIMO-GS shape NMSE.  Inside that band the ordering
maximizes the diversity of the GROUND-TRUTH dominant beam pair: farthest-point
sampling in the (Rx, Tx) beam-index plane, each axis normalized by its own beam
count so a 16-bin Rx shift counts as much as a 64-bin Tx shift.  The first
locations of that ordering therefore differ from each other in WHERE the
dominant lobe sits, which is what makes three panels next to each other
informative.

Style
-----
Mirrors ``evaluation/eval_measured.py``'s final-figure mode: two rows
(ground truth / MIMO-GS), one column per chosen location, linear scale
(per-map max normalization, shared 0..1 colorbar) plus a dB version
(``10*log10`` of the max-normalized map, floored at -30 dB), no suptitle,
axis labels at 14 pt, ticks at 12 pt, PNG at 300 dpi plus PDF.  The maps are
16 x 64, so the final-figure panels are wide and short: they are drawn with
``aspect="equal"``, which makes every map cell square and every panel 1:4, and
the figure height is computed from that.  Every panel in a figure has the same
size.  The gallery keeps its own ``aspect="auto"`` panels.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Import plumbing
# ---------------------------------------------------------------------------
# The evaluation scripts import repo-root packages (``scene``, ``arguments``,
# ``utils``) as top-level modules AND import each other as top-level modules,
# so both directories have to be importable -- the arrangement
# ``evaluation/eval_density.py`` already relies on.  This script lives at the
# repository root, so ``REPO_ROOT`` here is the real repository root; the
# imported modules compute their own (they sit one level down), which is why
# every path below is built locally instead of taken from ``eval_density``.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
EVALUATION_DIR = os.path.join(REPO_ROOT, "evaluation")

for _entry in (EVALUATION_DIR, REPO_ROOT):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import eval_render as ER  # noqa: E402  (path set up above)
from eval_density import (  # noqa: E402
    AXIS_LABEL_FONTSIZE,
    FIGURE_DPI,
    METHOD_MIMOGS,
    METHOD_MLP,
    TICK_LABELSIZE,
    TestGroundTruth,
    assert_finite_nonnegative,
    load_mimogs,
    load_mlp,
    predict_mlp_maps,
    render_mimogs_maps,
    resolve_device,
    write_csv,
    write_readme,
)
from utils.loss import normalize_mag_map  # noqa: E402


# ---------------------------------------------------------------------------
# Fixed inputs -- D5 is a single-configuration figure, so nothing is discovered
# ---------------------------------------------------------------------------
DEFAULT_DATASET_DIR = os.path.join(REPO_ROOT, "dataset", "asu_campus_16by64_lt")
DEFAULT_MIMOGS_CKPT = os.path.join(
    REPO_ROOT, "outputs", "density", "mimogs", "model_100.pth"
)
DEFAULT_MLP_CKPT = os.path.join(REPO_ROOT, "outputs", "density", "MLP", "model_100.pth")
DEFAULT_ANALYSIS_ROOT = os.path.join(REPO_ROOT, "analysis")
OUTPUT_NAME = "eval_samples"

METHOD_GT = "Ground truth"
ROW_ORDER: Tuple[str, str] = (METHOD_GT, METHOD_MIMOGS)

# Reference records the full-test-set self-check is compared against.
DENSITY_CSV = os.path.join("eval_density", "density_metrics.csv")
T1_TABLE_CSV = os.path.join("eval_t1", "t1_table.csv")
T1_PER_LOCATION_CSV = os.path.join("eval_t1", "t1_per_location.csv")
SPOTS_CANDIDATES_CSV = os.path.join("eval_spots", "candidates.csv")
SELF_CHECK_TOLERANCE_DB = 0.05

# Candidate band and gallery size.
BAND_LOW_PERCENTILE = 25.0
BAND_HIGH_PERCENTILE = 75.0
GALLERY_TOP = 20
PRINTED_CANDIDATES = 20

# Figure geometry.  The final figure draws every map cell SQUARE, so a panel is
# as wide as it always was and its HEIGHT follows the map shape: a 16 x 64 map
# gives a 1:4 panel.  PANEL_MARGIN_H_FINAL is the vertical space left for the
# column titles, the x label, the tick labels and the constrained-layout
# padding; it sits above the point where the figure HEIGHT -- rather than the
# figure width -- would start to limit the panels, so the drawn panel width is
# the same as before (3.23 in per column at three columns).
DB_FLOOR = -30.0
PANEL_W_INCH = 3.20
PANEL_MARGIN_W = 1.70
PANEL_MARGIN_H_FINAL = 1.00
# Square cells make a panel 0.80 in tall, but the vertical row label centred on
# it is taller than that ("Ground truth" measures 1.26 in at 14 pt), so it
# overhangs its panel at both ends and the two row labels would touch.  This is
# the extra figure height per row boundary that opens the gap between the rows
# to 0.37 in, which leaves about 0.10 in of clear space between the labels; the
# panels stay 3.22 in wide because the figure width is untouched.
ROW_LABEL_SLACK_INCH = 0.20
GALLERY_W_INCH = 8.20
GALLERY_H_INCH = 1.70
GALLERY_MARGIN_H = 1.35
COLUMN_TITLE_FONTSIZE = 12
COLORBAR_LABEL_FONTSIZE = 10
COLORBAR_TICK_LABELSIZE = 9
ANNOTATION_FONTSIZE = 9


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
def collect_predictions(
    arguments: argparse.Namespace, device: torch.device
) -> Dict[str, object]:
    """Render MIMO-GS and predict the MLP over the whole original test set."""
    mimogs_path = os.path.abspath(arguments.mimogs_ckpt)
    mlp_path = os.path.abspath(arguments.mlp_ckpt)
    dataset_dir = os.path.abspath(arguments.dataset or DEFAULT_DATASET_DIR)

    ground_truth = TestGroundTruth(dataset_dir, device)
    valid = torch.as_tensor(ground_truth.valid_indices, device=device)

    loaded_gs = load_mimogs(mimogs_path, device, dataset_dir)
    gs_maps = render_mimogs_maps(
        loaded_gs, ground_truth.positions_normalized, batch_size=arguments.batch_size
    )
    assert_finite_nonnegative(gs_maps, "MIMO-GS model_100")
    gs_scored = ground_truth.score(gs_maps)

    loaded_mlp = load_mlp(mlp_path, device)
    mlp_maps = predict_mlp_maps(loaded_mlp, ground_truth.positions_normalized)
    assert_finite_nonnegative(mlp_maps, "MLP model_100")
    mlp_scored = ground_truth.score(mlp_maps)

    maps: Dict[str, torch.Tensor] = {
        METHOD_GT: ground_truth.magnitude[valid],
        METHOD_MIMOGS: gs_maps[valid],
    }
    assert_finite_nonnegative(maps[METHOD_GT], "Ground truth")
    assert_finite_nonnegative(maps[METHOD_MIMOGS], "MIMO-GS drawn panels")

    return {
        "dataset_dir": dataset_dir,
        "ground_truth": ground_truth,
        "maps": maps,
        "scored": {METHOD_MIMOGS: gs_scored, METHOD_MLP: mlp_scored},
        "checkpoints": {
            METHOD_MIMOGS: os.path.relpath(mimogs_path, REPO_ROOT),
            METHOD_MLP: os.path.relpath(mlp_path, REPO_ROOT),
        },
        "num_gaussians": int(loaded_gs.num_gaussians),
        "beam_rows": int(ground_truth.beam_rows),
        "beam_cols": int(ground_truth.beam_cols),
    }


# ---------------------------------------------------------------------------
# Self-check against the recorded full-test-set numbers
# ---------------------------------------------------------------------------
def read_reference_nmse(analysis_root: str) -> Tuple[Optional[float], str]:
    """MIMO-GS model_100 mean shape NMSE as already recorded on disk."""
    for row in ER.read_csv_rows(os.path.join(analysis_root, DENSITY_CSV)):
        if row.get("method", "").strip() != METHOD_MIMOGS:
            continue
        if abs((ER._as_float(row.get("fraction")) or 0.0) - 1.0) > 1e-9:
            continue
        value = ER._as_float(row.get("nmse_shape_mean_dB"))
        if value is not None:
            return value, DENSITY_CSV

    for row in ER.read_csv_rows(os.path.join(analysis_root, T1_TABLE_CSV)):
        if row.get("method", "").strip() != METHOD_MIMOGS:
            continue
        value = ER._as_float(row.get("nmse_mean_dB"))
        if value is not None:
            return value, T1_TABLE_CSV

    return None, ""


def cross_check_per_location(
    analysis_root: str, predictions: Dict[str, object]
) -> List[str]:
    """Compare this run's per-location NMSE against the recorded tables.

    ``analysis/eval_spots/candidates.csv`` holds only the 30 locations that
    script selected, and no other file under ``analysis/eval_spots/`` carries a
    full per-location table, so the quality band cannot be READ from there --
    it needs all scored locations.  The overlapping rows are therefore used the
    way they can be used: as a check that this script's numbers are the same
    numbers, together with the full 3947-row table T1 already wrote.
    """
    ground_truth: TestGroundTruth = predictions["ground_truth"]
    position_of_test_index = {
        int(value): row
        for row, value in enumerate(ground_truth.valid_indices.tolist())
    }
    notes: List[str] = []

    sources = (
        (
            SPOTS_CANDIDATES_CSV,
            {METHOD_MIMOGS: "nmse_shape_dB_mimogs", METHOD_MLP: "nmse_shape_dB_mlp"},
        ),
        (
            T1_PER_LOCATION_CSV,
            {
                METHOD_MIMOGS: "nmse_shape_dB_MIMO-GS",
                METHOD_MLP: "nmse_shape_dB_MLP",
            },
        ),
    )

    for relative_path, columns in sources:
        rows = ER.read_csv_rows(os.path.join(analysis_root, relative_path))
        if not rows:
            notes.append(f"{relative_path}: not on disk, cross-check skipped")
            continue

        deltas: Dict[str, List[float]] = {method: [] for method in columns}
        for record in rows:
            test_index = ER._as_float(record.get("test_index"))
            if test_index is None:
                continue
            row = position_of_test_index.get(int(test_index))
            if row is None:
                continue
            for method, column in columns.items():
                recorded = ER._as_float(record.get(column))
                if recorded is None:
                    continue
                here = float(predictions["scored"][method]["nmse_shape_db"][row])
                deltas[method].append(abs(here - recorded))

        shared = max(len(values) for values in deltas.values())
        if shared == 0:
            notes.append(f"{relative_path}: no comparable row, cross-check skipped")
            continue
        worst = max(
            (max(values) if values else 0.0) for values in deltas.values()
        )
        notes.append(
            f"{relative_path}: {shared} shared location(s), "
            f"max |delta| = {worst:.4f} dB"
        )

    return notes


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------
def dominant_beam_pair(maps: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Per-location ``argmax`` of a ``(B, Nr, Nt)`` stack.

    Returns the flat beam-pair index and the ``(B, 2)`` ``(rx, tx)`` pair.
    """
    beam_cols = int(maps.shape[2])
    flat_index = (
        torch.argmax(maps.reshape(int(maps.shape[0]), -1), dim=1)
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    return flat_index, np.stack(
        [flat_index // beam_cols, flat_index % beam_cols], axis=1
    ).astype(np.int64)


def select_candidates(predictions: Dict[str, object]) -> Dict[str, object]:
    """Quality band, ordered for maximum dominant-beam-pair diversity."""
    ground_truth: TestGroundTruth = predictions["ground_truth"]
    shape_db = np.asarray(
        predictions["scored"][METHOD_MIMOGS]["nmse_shape_db"], dtype=np.float64
    )

    low = float(np.percentile(shape_db, BAND_LOW_PERCENTILE))
    high = float(np.percentile(shape_db, BAND_HIGH_PERCENTILE))
    band = np.flatnonzero((shape_db >= low) & (shape_db <= high)).astype(np.int64)
    if band.size == 0:
        raise SystemExit("[eval_samples] The quality band is empty; nothing to draw.")

    flat_index, pairs = dominant_beam_pair(predictions["maps"][METHOD_GT])

    # Farthest-point sampling in the (Rx, Tx) beam-index plane.  Each axis is
    # divided by its own beam count, so the 16 Rx bins and the 64 Tx bins span
    # the same range and a shift of one full aperture counts the same on both
    # sides.  Ties (identical distance) go to the better-rendered location, then
    # to the lower test index, so the ordering is deterministic.
    scale = np.array(
        [float(predictions["beam_rows"]), float(predictions["beam_cols"])],
        dtype=np.float64,
    )
    coordinates = pairs[band].astype(np.float64) / scale
    quality = shape_db[band]
    test_index = ground_truth.valid_indices[band].astype(np.int64)

    # Seed with the most typical location of the band: the one whose NMSE is
    # closest to the band median.
    seed = int(np.argmin(np.abs(quality - float(np.median(quality)))))

    remaining = np.ones(band.size, dtype=bool)
    order = [seed]
    diversity_score = np.zeros(band.size, dtype=np.float64)
    remaining[seed] = False
    min_distance = np.linalg.norm(coordinates - coordinates[seed], axis=1)
    diversity_score[seed] = float("inf")

    while bool(remaining.any()):
        best = float(min_distance[remaining].max())
        pool = np.flatnonzero(remaining & (min_distance >= best - 1e-12))
        pick = int(pool[np.lexsort((test_index[pool], quality[pool]))[0]])
        diversity_score[pick] = float(min_distance[pick])
        order.append(pick)
        remaining[pick] = False
        min_distance = np.minimum(
            min_distance, np.linalg.norm(coordinates - coordinates[pick], axis=1)
        )

    ordered = np.asarray(order, dtype=np.int64)
    return {
        "band_rows": band[ordered],
        "band_low_db": low,
        "band_high_db": high,
        "band_size": int(band.size),
        "diversity_score": diversity_score[ordered],
        "gt_flat_index": flat_index,
        "gt_pairs": pairs,
        "num_unique_pairs_band": int(np.unique(pairs[band], axis=0).shape[0]),
    }


def gallery_rows(candidates: Dict[str, object], how_many: int) -> List[int]:
    rows = [int(row) for row in candidates["band_rows"]]
    return rows if how_many <= 0 else rows[: int(how_many)]


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def to_linear_panel(single_map: torch.Tensor) -> np.ndarray:
    """Max-normalized map in [0, 1]."""
    return normalize_mag_map(single_map.unsqueeze(0))[0].detach().cpu().numpy()


def to_db_panel(single_map: torch.Tensor) -> np.ndarray:
    """``10*log10`` of the max-normalized map, floored at ``DB_FLOOR``."""
    linear = to_linear_panel(single_map)
    with np.errstate(divide="ignore"):
        decibel = 10.0 * np.log10(np.maximum(linear, 1e-12))
    return np.maximum(decibel, DB_FLOOR)


def panel_limits(scale: str) -> Tuple[float, float, str]:
    if scale == "linear":
        return 0.0, 1.0, "Normalized power"
    return DB_FLOOR, 0.0, "Normalized power [dB]"


def panel_converter(scale: str):
    return to_linear_panel if scale == "linear" else to_db_panel


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------
def render_gallery_figure(
    output_dir: str,
    row: int,
    predictions: Dict[str, object],
    candidates: Dict[str, object],
    scale: str,
) -> str:
    """One candidate: ground truth over MIMO-GS, plus the picking annotation."""
    ground_truth: TestGroundTruth = predictions["ground_truth"]
    vmin, vmax, colorbar_label = panel_limits(scale)
    convert = panel_converter(scale)

    test_index = int(ground_truth.valid_indices[row])
    position = ground_truth.valid_positions_m[row]
    pair = candidates["gt_pairs"][row]

    figure, axes = plt.subplots(
        len(ROW_ORDER),
        1,
        figsize=(
            GALLERY_W_INCH,
            GALLERY_H_INCH * len(ROW_ORDER) + GALLERY_MARGIN_H,
        ),
        squeeze=False,
        layout="constrained",
    )

    image = None
    for panel, method in enumerate(ROW_ORDER):
        axis = axes[panel][0]
        image = axis.imshow(
            convert(predictions["maps"][method][row]),
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        axis.set_ylabel(method, fontsize=AXIS_LABEL_FONTSIZE)
        axis.tick_params(labelsize=TICK_LABELSIZE)
        if panel == len(ROW_ORDER) - 1:
            axis.set_xlabel("Tx beam index", fontsize=AXIS_LABEL_FONTSIZE)
        else:
            axis.set_xticklabels([])

    # Above the panels rather than inside them: an overlay box would cover the
    # first beam rows of the ground-truth map, which are real data.
    figure.suptitle(
        f"test index {test_index}   "
        f"(x, y, z) = ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) m   "
        f"GT top-1 beam pair (rx, tx) = ({int(pair[0])}, {int(pair[1])})\n"
        f"shape NMSE: MIMO-GS "
        f"{float(predictions['scored'][METHOD_MIMOGS]['nmse_shape_db'][row]):.2f} dB   "
        f"MLP {float(predictions['scored'][METHOD_MLP]['nmse_shape_db'][row]):.2f} dB",
        fontsize=ANNOTATION_FONTSIZE,
    )

    colorbar = figure.colorbar(
        image,
        ax=[axis for row_axes in axes for axis in row_axes],
        fraction=0.030,
        pad=0.012,
    )
    colorbar.set_label(colorbar_label, fontsize=COLORBAR_LABEL_FONTSIZE)
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_LABELSIZE)

    target_dir = os.path.join(output_dir, "gallery", scale)
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, f"loc_{test_index}.png")
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# Final publication figure
# ---------------------------------------------------------------------------
def render_sample_grid(
    output_dir: str,
    rows: Sequence[int],
    predictions: Dict[str, object],
    scale: str,
) -> Tuple[str, str]:
    """The final figure: chosen locations as columns, two map rows.

    Ground truth over MIMO-GS, one shared colorbar, no suptitle -- the caption
    of the paper carries the location identity, and the columns are named
    "Spot k" so the text can refer to them.
    """
    vmin, vmax, colorbar_label = panel_limits(scale)
    convert = panel_converter(scale)

    # Square cells: a panel that is PANEL_W_INCH wide is beam_rows / beam_cols
    # as tall (0.80 in for the 16 x 64 maps), and the figure height is sized to
    # that so ``aspect="equal"`` does not have to shrink the panels to fit.
    panel_h_inch = PANEL_W_INCH * (
        int(predictions["beam_rows"]) / int(predictions["beam_cols"])
    )

    figure, axes = plt.subplots(
        len(ROW_ORDER),
        len(rows),
        figsize=(
            PANEL_W_INCH * len(rows) + PANEL_MARGIN_W,
            panel_h_inch * len(ROW_ORDER)
            + ROW_LABEL_SLACK_INCH * (len(ROW_ORDER) - 1)
            + PANEL_MARGIN_H_FINAL,
        ),
        squeeze=False,
        layout="constrained",
    )

    image = None
    for column, row in enumerate(rows):
        for panel, method in enumerate(ROW_ORDER):
            axis = axes[panel][column]
            image = axis.imshow(
                convert(predictions["maps"][method][row]),
                aspect="equal",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
            )

            axis.tick_params(labelsize=TICK_LABELSIZE)
            if panel == 0:
                axis.set_title(f"Spot {column + 1}", fontsize=COLUMN_TITLE_FONTSIZE)
            if column == 0:
                axis.set_ylabel(method, fontsize=AXIS_LABEL_FONTSIZE)
            else:
                axis.set_yticklabels([])
            if panel == len(ROW_ORDER) - 1:
                axis.set_xlabel("Tx beam index", fontsize=AXIS_LABEL_FONTSIZE)
            else:
                axis.set_xticklabels([])

    colorbar = figure.colorbar(
        image,
        ax=[axis for row_axes in axes for axis in row_axes],
        fraction=0.030,
        pad=0.012,
    )
    # The colorbar is a labelled axis like any other in this figure, so its
    # label and ticks follow the axis convention (14 pt / 12 pt) rather than the
    # smaller sizes the gallery uses.
    colorbar.set_label(colorbar_label, fontsize=AXIS_LABEL_FONTSIZE)
    colorbar.ax.tick_params(labelsize=TICK_LABELSIZE)

    os.makedirs(output_dir, exist_ok=True)
    stem = f"fig_deepmimo_samples_{scale}"
    png_path = os.path.join(output_dir, f"{stem}.png")
    pdf_path = os.path.join(output_dir, f"{stem}.pdf")
    figure.savefig(png_path, dpi=FIGURE_DPI)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
CANDIDATE_COLUMNS = (
    "rank",
    "test_index",
    "x_m",
    "y_m",
    "z_m",
    "nmse_shape_dB_mimogs",
    "nmse_shape_dB_mlp",
    "gt_top1_flat_index",
    "gt_top1_rx",
    "gt_top1_tx",
    "diversity_distance",
    "in_gallery",
)


def candidate_records(
    predictions: Dict[str, object],
    candidates: Dict[str, object],
    drawn: Sequence[int],
) -> List[List[object]]:
    ground_truth: TestGroundTruth = predictions["ground_truth"]
    drawn_set = set(int(row) for row in drawn)
    records: List[List[object]] = []
    for rank, row in enumerate(candidates["band_rows"]):
        row = int(row)
        position = ground_truth.valid_positions_m[row]
        pair = candidates["gt_pairs"][row]
        distance = float(candidates["diversity_score"][rank])
        records.append(
            [
                rank + 1,
                int(ground_truth.valid_indices[row]),
                f"{position[0]:.6f}",
                f"{position[1]:.6f}",
                f"{position[2]:.6f}",
                f"{float(predictions['scored'][METHOD_MIMOGS]['nmse_shape_db'][row]):.6f}",
                f"{float(predictions['scored'][METHOD_MLP]['nmse_shape_db'][row]):.6f}",
                int(candidates["gt_flat_index"][row]),
                int(pair[0]),
                int(pair[1]),
                "" if not np.isfinite(distance) else f"{distance:.6f}",
                int(row in drawn_set),
            ]
        )
    return records


def print_candidate_table(
    predictions: Dict[str, object],
    candidates: Dict[str, object],
    how_many: int,
) -> None:
    ground_truth: TestGroundTruth = predictions["ground_truth"]
    rows = [int(row) for row in candidates["band_rows"][: int(how_many)]]

    print("")
    print("-" * 104)
    print(
        f"[eval_samples] CANDIDATES -- MIMO-GS shape NMSE between the "
        f"{BAND_LOW_PERCENTILE:.0f}th ({float(candidates['band_low_db']):.2f} dB) "
        f"and the {BAND_HIGH_PERCENTILE:.0f}th "
        f"({float(candidates['band_high_db']):.2f} dB) percentile"
    )
    print(
        f"[eval_samples] {int(candidates['band_size'])} locations in the band, "
        f"ordered for maximum GT dominant-beam-pair diversity; "
        f"first {len(rows)} shown"
    )
    print("-" * 104)
    header = (
        f"  {'rank':>5}{'test idx':>10}{'x [m]':>10}{'y [m]':>10}"
        f"{'GS [dB]':>10}{'MLP [dB]':>10}"
        f"{'GT top-1':>11}{'(rx, tx)':>12}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rank, row in enumerate(rows):
        position = ground_truth.valid_positions_m[row]
        pair = candidates["gt_pairs"][row]
        print(
            f"  {rank + 1:>5}"
            f"{int(ground_truth.valid_indices[row]):>10}"
            f"{position[0]:>10.2f}{position[1]:>10.2f}"
            f"{float(predictions['scored'][METHOD_MIMOGS]['nmse_shape_db'][row]):>10.2f}"
            f"{float(predictions['scored'][METHOD_MLP]['nmse_shape_db'][row]):>10.2f}"
            f"{int(candidates['gt_flat_index'][row]):>11}"
            f"{f'({int(pair[0])}, {int(pair[1])})':>12}"
        )
    print("  " + "-" * (len(header) - 2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_spot_argument(text: str) -> List[int]:
    tokens = [token for token in str(text).replace(",", " ").split() if token]
    values: List[int] = []
    for token in tokens:
        try:
            values.append(int(token))
        except ValueError as error:
            raise SystemExit(
                f"[eval_samples] --spots value is not an integer: {token!r}"
            ) from error
    if not 2 <= len(values) <= 4:
        raise SystemExit(
            f"[eval_samples] --spots takes 2 to 4 test indices, got {len(values)}."
        )
    if len(set(values)) != len(values):
        raise SystemExit("[eval_samples] --spots holds a duplicated test index.")
    return values


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="D5 -- DeepMIMO qualitative rendering samples"
    )
    parser.add_argument(
        "--mimogs_ckpt",
        type=str,
        default=DEFAULT_MIMOGS_CKPT,
        help=f"MIMO-GS density repack (default: "
        f"{os.path.relpath(DEFAULT_MIMOGS_CKPT, REPO_ROOT)})",
    )
    parser.add_argument(
        "--mlp_ckpt",
        type=str,
        default=DEFAULT_MLP_CKPT,
        help=f"MLP density repack, listed for reference only (default: "
        f"{os.path.relpath(DEFAULT_MLP_CKPT, REPO_ROOT)})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help=f"Override the dataset directory (default: "
        f"{os.path.relpath(DEFAULT_DATASET_DIR, REPO_ROOT)})",
    )
    parser.add_argument("--analysis_root", type=str, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Rendering batch size for the MIMO-GS forward passes.",
    )
    parser.add_argument(
        "--spots",
        type=str,
        default="",
        help="Final-figure mode: 2 to 4 comma-separated test indices, "
        "e.g. --spots 12,34,56",
    )
    parser.add_argument(
        "--gallery",
        action="store_true",
        help="Also render the candidate gallery when --spots is given.",
    )
    parser.add_argument(
        "--gallery_top",
        type=int,
        default=GALLERY_TOP,
        help=f"How many of the diverse candidates to draw (default {GALLERY_TOP}; "
        f"0 draws the whole band).",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    return parser


def main() -> int:
    arguments = build_argument_parser().parse_args()
    device = resolve_device(prefer_cuda=not arguments.cpu)
    warnings: List[str] = []

    print("=" * 104)
    print("[eval_samples] DeepMIMO qualitative rendering samples")
    print("=" * 104)
    print(
        f"[eval_samples] device : {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    )

    # --spots is validated before any model is loaded, so a typo fails fast.
    requested_spots = parse_spot_argument(arguments.spots) if arguments.spots else []

    predictions = collect_predictions(arguments, device)
    ground_truth: TestGroundTruth = predictions["ground_truth"]

    print(f"[eval_samples] dataset: {predictions['dataset_dir']}")
    print(
        f"[eval_samples] maps   : {predictions['beam_rows']} x "
        f"{predictions['beam_cols']} (Rx x Tx)"
    )
    print(
        f"[eval_samples] scored test locations: {ground_truth.num_scored} "
        f"(skipped zero-power: {ground_truth.num_skipped_zero_power})"
    )
    print(
        f"[eval_samples] MIMO-GS: {predictions['checkpoints'][METHOD_MIMOGS]} "
        f"({predictions['num_gaussians']} Gaussians)"
    )
    print(f"[eval_samples] MLP    : {predictions['checkpoints'][METHOD_MLP]}")

    # -- sanity ------------------------------------------------------------
    measured = float(np.mean(predictions["scored"][METHOD_MIMOGS]["nmse_shape_db"]))
    reference, reference_source = read_reference_nmse(arguments.analysis_root)

    print("")
    print("-" * 104)
    print("[eval_samples] SANITY -- MIMO-GS model_100, full test set")
    print("-" * 104)
    print(f"  this script                : {measured:.4f} dB")
    delta: Optional[float] = None
    if reference is None:
        print("  recorded reference         : not found "
              f"({DENSITY_CSV} / {T1_TABLE_CSV})")
        warnings.append(
            "WARN self-check skipped: neither analysis/eval_density/"
            "density_metrics.csv nor analysis/eval_t1/t1_table.csv is on disk."
        )
    else:
        delta = abs(measured - float(reference))
        status = "ok" if delta <= SELF_CHECK_TOLERANCE_DB else "MISMATCH"
        print(f"  recorded reference         : {float(reference):.4f} dB "
              f"({reference_source})")
        print(f"  delta                      : {delta:.4f} dB "
              f"(tolerance {SELF_CHECK_TOLERANCE_DB:.2f} dB) -> {status}")
        if status != "ok":
            warnings.append(
                f"WARN self-check MISMATCH: {measured:.4f} dB here vs. "
                f"{float(reference):.4f} dB in {reference_source} "
                f"(delta {delta:.4f} dB)."
            )

    cross_notes = cross_check_per_location(arguments.analysis_root, predictions)
    for note in cross_notes:
        print(f"  per-location cross-check   : {note}")
    print(
        f"  panels finite, nonnegative : yes "
        f"(min GT {float(predictions['maps'][METHOD_GT].min()):.3g}, "
        f"min MIMO-GS {float(predictions['maps'][METHOD_MIMOGS].min()):.3g})"
    )

    # -- candidates --------------------------------------------------------
    candidates = select_candidates(predictions)
    output_dir = os.path.join(arguments.analysis_root, OUTPUT_NAME)
    os.makedirs(output_dir, exist_ok=True)

    print_candidate_table(predictions, candidates, PRINTED_CANDIDATES)

    # -- figures -----------------------------------------------------------
    spot_rows: List[int] = []
    if requested_spots:
        lookup = {
            int(value): row
            for row, value in enumerate(ground_truth.valid_indices.tolist())
        }
        missing = [value for value in requested_spots if value not in lookup]
        if missing:
            raise SystemExit(
                f"[eval_samples] --spots refers to test indices that are not "
                f"scored: {missing}"
            )
        spot_rows = [lookup[value] for value in requested_spots]

    drawn: List[int] = []
    if not requested_spots or arguments.gallery:
        drawn = gallery_rows(candidates, arguments.gallery_top)
        print("")
        print(
            f"[eval_samples] rendering the gallery "
            f"({len(drawn)} locations x 2 scales)..."
        )
        for row in drawn:
            for scale in ("linear", "db"):
                render_gallery_figure(output_dir, row, predictions, candidates, scale)
        print(
            f"[eval_samples] gallery written to "
            f"{os.path.join(os.path.relpath(output_dir, REPO_ROOT), 'gallery', '{linear,db}')}"
        )

    written: List[str] = []
    if spot_rows:
        print("")
        print(
            f"[eval_samples] final-figure mode: test indices "
            f"{[int(ground_truth.valid_indices[row]) for row in spot_rows]}"
        )
        for scale in ("linear", "db"):
            written.extend(render_sample_grid(output_dir, spot_rows, predictions, scale))
        for path in written:
            print(f"[eval_samples]   {os.path.relpath(path, REPO_ROOT)}")

    write_csv(
        os.path.join(output_dir, "candidates.csv"),
        CANDIDATE_COLUMNS,
        candidate_records(predictions, candidates, drawn),
    )

    # -- README ------------------------------------------------------------
    unique_drawn = (
        int(np.unique(candidates["gt_pairs"][np.asarray(drawn, dtype=np.int64)], axis=0).shape[0])
        if drawn
        else 0
    )
    readme = [
        "eval_samples -- DeepMIMO qualitative rendering samples",
        "=" * 70,
        "",
        "CONVENTIONS",
        f"  Model       : {predictions['checkpoints'][METHOD_MIMOGS]} "
        f"({predictions['num_gaussians']} Gaussians)",
        f"  Reference   : {predictions['checkpoints'][METHOD_MLP]} "
        f"(numbers only, never drawn)",
        "  Metric      : shape NMSE (max-normalized prediction vs. max-normalized",
        "                target) per location, in dB.  Imported from",
        "                evaluation/eval_render.py; never reimplemented.",
        f"  Test set    : the original full test.mat of {predictions['dataset_dir']};",
        f"                {ground_truth.num_scored} locations scored, maps are "
        f"{predictions['beam_rows']} x {predictions['beam_cols']} (Rx x Tx).",
        f"  Band        : MIMO-GS shape NMSE between the "
        f"{BAND_LOW_PERCENTILE:.0f}th ({float(candidates['band_low_db']):.2f} dB)",
        f"                and the {BAND_HIGH_PERCENTILE:.0f}th "
        f"({float(candidates['band_high_db']):.2f} dB) percentile -- typical",
        f"                rendering behaviour, not the tails.  "
        f"{int(candidates['band_size'])} locations,",
        f"                {int(candidates['num_unique_pairs_band'])} distinct GT "
        f"dominant beam pairs.",
        "  Ordering    : farthest-point sampling of the GT dominant beam pair in the",
        "                (Rx, Tx) index plane, each axis divided by its beam count;",
        "                seeded at the band-median location, ties to the better NMSE.",
        "  Scales      : 'linear' = each map divided by its own max, shared colorbar",
        f"                0..1.  'db' = 10*log10 of that, floored at {DB_FLOOR:.0f} dB.",
        f"  Device      : {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
        "  Figures     : final figure has no suptitle; column titles 'Spot k' at "
        f"{COLUMN_TITLE_FONTSIZE} pt,",
        f"                axis labels {AXIS_LABEL_FONTSIZE} pt, ticks "
        f"{TICK_LABELSIZE} pt, PNG at {FIGURE_DPI} dpi plus PDF.",
        "",
        "SANITY",
        f"  MIMO-GS model_100, full test set, this script : {measured:.4f} dB",
    ]
    if reference is None:
        readme.append("  recorded reference                           : not found")
    else:
        readme += [
            f"  recorded reference ({reference_source:<28}): {float(reference):.4f} dB",
            f"  delta                                        : {float(delta):.4f} dB "
            f"(tolerance {SELF_CHECK_TOLERANCE_DB:.2f} dB)",
        ]
    readme += ["  per-location cross-checks:"] + [f"    {note}" for note in cross_notes]
    readme += [
        "",
        "DRAWN",
        f"  gallery locations : {len(drawn)}"
        + (f" ({unique_drawn} distinct GT dominant beam pairs)" if drawn else ""),
        "  final-figure spots: "
        + str([int(ground_truth.valid_indices[row]) for row in spot_rows] or "not requested"),
        "",
        "FILES",
        "  candidates.csv                       the whole band in diversity order",
        "  gallery/linear/loc_<index>.png       rows = GT / MIMO-GS, linear scale",
        "  gallery/db/loc_<index>.png           same, dB scale",
        "  fig_deepmimo_samples_{linear,db}.*   final figure (--spots mode only)",
        "  README.txt                           this file",
        "",
        "WARNINGS",
    ]
    readme += [f"  {warning}" for warning in warnings] or ["  none"]
    readme += [
        "",
        "RERUN",
        "  python eval_samples.py",
        "  python eval_samples.py --spots <i>,<j>,<k>",
    ]
    write_readme(os.path.join(output_dir, "README.txt"), readme)

    print("")
    print("=" * 104)
    print("[eval_samples] SUMMARY")
    print("=" * 104)
    print(f"  mean shape NMSE, MIMO-GS : {measured:.3f} dB "
          f"({ground_truth.num_scored} locations)")
    print(f"  mean shape NMSE, MLP     : "
          f"{float(np.mean(predictions['scored'][METHOD_MLP]['nmse_shape_db'])):.3f} dB")
    print(f"  band                     : {int(candidates['band_size'])} locations, "
          f"{float(candidates['band_low_db']):.2f} .. "
          f"{float(candidates['band_high_db']):.2f} dB")
    print(f"  gallery figures          : {2 * len(drawn)}")
    print(f"  final-figure spots       : "
          f"{[int(ground_truth.valid_indices[row]) for row in spot_rows] or 'not requested'}")
    print("")
    if warnings:
        print(f"[eval_samples] {len(warnings)} WARNING(S):")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("[eval_samples] No warnings.")
    print(f"[eval_samples] Outputs written to {os.path.relpath(output_dir, REPO_ROOT)}")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    sys.exit(main())
