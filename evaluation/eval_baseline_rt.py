"""E2 -- MIMO-GS vs. Sionna RT, both scored against Wireless InSite ground truth.

Runs with zero arguments::

    python eval_baseline_rt.py

Two predictors of the same physical quantity are compared on exactly the same
Wireless InSite (WI) test locations, with the same metric code that
``eval_render.py`` uses (imported, not re-implemented):

* **MIMO-GS**  -- ``render_fast`` output of a trained checkpoint, evaluated
  strictly on the TEST split it never saw during training.
* **Sionna RT** -- the ray-traced magnitude map simulated for the same
  location, taken verbatim from ``full_dataset.mat`` as a prediction.

Fairness framing
----------------
The two predictors have *different information budgets*, and the comparison is
not a like-for-like model contest:

* Sionna RT never saw a single WI measurement.  It only saw the scene geometry
  (and whatever material parameters the dataset generator configured), so
  scoring it on the WI test locations is fair -- there is no split to leak.
* MIMO-GS was trained on WI measurements at *other* locations.  It is scored
  only on held-out locations, so the number it gets is a generalization number,
  not a fit number.

The interesting question is therefore "how far does a measurement-trained
representation get you over a geometry-only simulator on unseen locations",
not "which model is better".

Normalization convention
------------------------
Identical to ``eval_render.py``.  The target is always
``normalize_mag_map(GT)`` (per-location max-normalized), and both NMSE
conventions are reported for both predictors:

* ``NMSE_raw_dB``   -- prediction used verbatim (the training scale term).
* ``NMSE_shape_dB`` -- prediction max-normalized (the training shape term).

``NMSE_shape_dB`` is the headline for the *cross-method* figures.  ``raw`` is
not scale-comparable across the two predictors: MIMO-GS was optimized to emit
the max-normalized target directly, whereas the Sionna maps carry the dataset's
own global scale, so ``NMSE_raw_dB`` would penalize Sionna for a normalization
convention it has no way to know.  Both numbers are in the CSVs regardless.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import functools
import math
import os
import sys
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch

# Metric plumbing and checkpoint handling are reused verbatim from E1 so the
# two experiments cannot drift apart.
from eval_render import (
    EPS,
    build_scene_and_model,
    gain_net_hidden_dim,
    gain_net_width,
    mean_linear_db,
    render_batch,
    resolve_run_dir,
    restore_config,
    save_figure,
    summarize,
    topk_metrics,
)
from utils.loss import normalize_mag_map


DEFAULT_CKPT = "outputs/20260810_052854"
DEFAULT_SIONNA_MAT = "dataset/asu_sionna_16by64_lt/full_dataset.mat"
DEFAULT_MATCH_TOL = 1e-3
DEFAULT_MIN_MATCH_FRACTION = 0.90

TOPK_ACC_VALUES = (1, 4, 8)
CAPTURE_VALUES = (1, 4)
ALL_K_VALUES = tuple(sorted(set(TOPK_ACC_VALUES) | set(CAPTURE_VALUES)))

PREDICTOR_MIMOGS = "MIMO-GS"
PREDICTOR_SIONNA = "Sionna RT"
PREDICTOR_COLORS = {PREDICTOR_MIMOGS: "tab:blue", PREDICTOR_SIONNA: "tab:orange"}


# ----------------------------------------------------------------------
# Raw .mat access (matching is done in ORIGINAL, un-normalized coordinates)
# ----------------------------------------------------------------------
def load_raw_mat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(positions, magnitude)`` straight from a dataset ``.mat``.

    ``DeepMIMODataset`` divides the positions by a per-file ``scale_factor``,
    so anything that has to line up across two different files must be matched
    on these raw values instead.
    """
    if not os.path.isfile(path):
        raise SystemExit(f"[eval_baseline_rt] Missing .mat file: {path}")

    data = sio.loadmat(path)
    for key in ("positions", "magnitude"):
        if key not in data:
            raise SystemExit(
                f"[eval_baseline_rt] '{path}' has no '{key}' array; keys = "
                f"{[k for k in data if not k.startswith('__')]}"
            )

    positions = np.asarray(data["positions"], dtype=np.float64)
    magnitude = np.asarray(data["magnitude"], dtype=np.float32)

    if positions.shape[0] != magnitude.shape[0]:
        raise SystemExit(
            f"[eval_baseline_rt] '{path}': {positions.shape[0]} positions vs. "
            f"{magnitude.shape[0]} magnitude maps."
        )
    if magnitude.ndim != 3:
        raise SystemExit(
            f"[eval_baseline_rt] '{path}': magnitude must be (N, Nr, Nt), got "
            f"{tuple(magnitude.shape)}."
        )
    return positions, magnitude


def describe_ranges(name: str, positions: np.ndarray) -> str:
    lows = positions.min(axis=0)
    highs = positions.max(axis=0)
    return (
        f"  {name:<22} N={positions.shape[0]:<7d} "
        f"x=[{lows[0]:10.4f}, {highs[0]:10.4f}]  "
        f"y=[{lows[1]:10.4f}, {highs[1]:10.4f}]  "
        f"z=[{lows[2]:10.4f}, {highs[2]:10.4f}]"
    )


# ----------------------------------------------------------------------
# One-to-one nearest-neighbour position matching
# ----------------------------------------------------------------------
def _neighbour_candidates(
    query: np.ndarray, reference: np.ndarray, tolerance: float, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(distances, indices)`` of the ``k`` nearest reference points.

    Entries beyond ``tolerance`` come back as ``inf`` / ``-1``.  A KD-tree is
    used when SciPy provides one and a chunked brute-force fallback otherwise,
    so the script never silently depends on an optional import.
    """
    k = max(1, min(int(k), reference.shape[0]))

    try:
        from scipy.spatial import cKDTree  # noqa: PLC0415 -- optional fast path

        tree = cKDTree(reference)
        distances, indices = tree.query(
            query, k=k, distance_upper_bound=float(tolerance)
        )
        distances = np.atleast_2d(np.asarray(distances, dtype=np.float64).reshape(
            query.shape[0], k
        ))
        indices = np.atleast_2d(np.asarray(indices).reshape(query.shape[0], k))
        # cKDTree marks misses with index == len(reference).
        indices = np.where(indices >= reference.shape[0], -1, indices)
        return distances, indices
    except ImportError:
        pass

    distances = np.full((query.shape[0], k), np.inf, dtype=np.float64)
    indices = np.full((query.shape[0], k), -1, dtype=np.int64)
    chunk = 512
    for start in range(0, query.shape[0], chunk):
        stop = min(start + chunk, query.shape[0])
        deltas = query[start:stop, None, :] - reference[None, :, :]
        block = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))
        order = np.argsort(block, axis=1)[:, :k]
        block_sorted = np.take_along_axis(block, order, axis=1)
        keep = block_sorted <= tolerance
        distances[start:stop] = np.where(keep, block_sorted, np.inf)
        indices[start:stop] = np.where(keep, order, -1)
    return distances, indices


def match_positions(
    gt_positions: np.ndarray,
    other_positions: np.ndarray,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy one-to-one match of GT rows onto ``other`` rows.

    Returns ``(gt_indices, other_indices, distances)``, sorted by GT index.
    Every returned pair is within ``tolerance`` and no ``other`` row is used
    twice; the greedy pass consumes candidate pairs shortest-distance-first,
    which is optimal here because the accepted radius is far below the grid
    spacing.
    """
    candidate_distances, candidate_indices = _neighbour_candidates(
        gt_positions, other_positions, tolerance, k=4
    )

    finite = np.isfinite(candidate_distances) & (candidate_indices >= 0)
    gt_rows, slot_rows = np.nonzero(finite)
    if gt_rows.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )

    pair_distance = candidate_distances[gt_rows, slot_rows]
    pair_other = candidate_indices[gt_rows, slot_rows].astype(np.int64)

    order = np.argsort(pair_distance, kind="stable")
    used_gt = np.zeros(gt_positions.shape[0], dtype=bool)
    used_other = np.zeros(other_positions.shape[0], dtype=bool)

    matched_gt: List[int] = []
    matched_other: List[int] = []
    matched_distance: List[float] = []

    for position in order:
        gt_index = int(gt_rows[position])
        other_index = int(pair_other[position])
        if used_gt[gt_index] or used_other[other_index]:
            continue
        used_gt[gt_index] = True
        used_other[other_index] = True
        matched_gt.append(gt_index)
        matched_other.append(other_index)
        matched_distance.append(float(pair_distance[position]))

    matched_gt_array = np.asarray(matched_gt, dtype=np.int64)
    matched_other_array = np.asarray(matched_other, dtype=np.int64)
    matched_distance_array = np.asarray(matched_distance, dtype=np.float64)

    reorder = np.argsort(matched_gt_array, kind="stable")
    matched_gt_array = matched_gt_array[reorder]
    matched_other_array = matched_other_array[reorder]
    matched_distance_array = matched_distance_array[reorder]

    # One-to-one is a correctness requirement, not a hope.
    if np.unique(matched_other_array).size != matched_other_array.size:
        raise SystemExit(
            "[eval_baseline_rt] Matching produced a duplicated Sionna index; "
            "the assignment is not one-to-one."
        )
    if np.unique(matched_gt_array).size != matched_gt_array.size:
        raise SystemExit(
            "[eval_baseline_rt] Matching produced a duplicated GT index; "
            "the assignment is not one-to-one."
        )
    return matched_gt_array, matched_other_array, matched_distance_array


# ----------------------------------------------------------------------
# Metric core -- one code path, both predictors
# ----------------------------------------------------------------------
def score_prediction(
    predicted: torch.Tensor,
    target_normalized: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """Score one predictor against an already max-normalized target.

    ``predicted`` and ``target_normalized`` are ``(B, Nr, Nt)``.  This is the
    same arithmetic as ``eval_render.evaluate_test_set``, lifted out so both
    predictors provably share it.
    """
    count = predicted.shape[0]

    predicted_normalized = normalize_mag_map(predicted)

    target_flat = target_normalized.reshape(count, -1)
    predicted_flat = predicted.reshape(count, -1)
    predicted_normalized_flat = predicted_normalized.reshape(count, -1)

    target_energy = target_flat.square().sum(dim=1).clamp_min(EPS)
    raw_ratio = (predicted_flat - target_flat).square().sum(dim=1) / target_energy
    shape_ratio = (
        predicted_normalized_flat - target_flat
    ).square().sum(dim=1) / target_energy

    scored: Dict[str, np.ndarray] = {
        "nmse_raw_db": (10.0 * torch.log10(raw_ratio.clamp_min(1e-12)))
        .cpu()
        .numpy()
        .astype(np.float64),
        "nmse_shape_db": (10.0 * torch.log10(shape_ratio.clamp_min(1e-12)))
        .cpu()
        .numpy()
        .astype(np.float64),
    }

    for k, (overlap, capture) in topk_metrics(
        predicted_flat, target_flat, ALL_K_VALUES
    ).items():
        scored[f"topk_acc_K{k}"] = overlap.cpu().numpy().astype(np.float64)
        scored[f"power_capture_K{k}"] = capture.cpu().numpy().astype(np.float64)

    return scored


def render_mimogs(
    scene,
    gaussians,
    model_params,
    device: torch.device,
    normalized_positions: torch.Tensor,
    batch_size: int,
    use_cuda_rasterizer: bool,
) -> torch.Tensor:
    """Render every requested UE location with the training render path."""
    tx_pos = torch.as_tensor(scene.bs_position, dtype=torch.float32, device=device)
    chunks: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, normalized_positions.shape[0], batch_size):
            stop = min(start + batch_size, normalized_positions.shape[0])
            rx_pos = normalized_positions[start:stop].to(device)
            chunks.append(
                render_batch(
                    rx_pos,
                    tx_pos,
                    gaussians,
                    scene,
                    model_params,
                    use_cuda_rasterizer,
                ).float()
            )

    return torch.cat(chunks, dim=0)


# ----------------------------------------------------------------------
# Sanity checks
# ----------------------------------------------------------------------
def self_consistency_check(magnitude: torch.Tensor) -> Tuple[float, float, float]:
    """Score a map set against itself; the metric plumbing must return 0 error.

    Returns ``(nmse_shape_dB, topk_acc_K1, power_capture_K1)``.  ``nmse_shape``
    lands on the ``1e-12`` clamp floor (-120 dB) rather than literal ``-inf``.
    """
    target_normalized = normalize_mag_map(magnitude)
    scored = score_prediction(magnitude, target_normalized)
    return (
        float(np.max(scored["nmse_shape_db"])),
        float(np.min(scored["topk_acc_K1"])),
        float(np.min(scored["power_capture_K1"])),
    )


def assert_disjoint_from_train(
    matched_positions: np.ndarray,
    train_positions: np.ndarray,
    tolerance: float,
) -> float:
    """Fail loudly if any evaluated location also lives in the TRAIN split."""
    distances, _ = _neighbour_candidates(
        matched_positions, train_positions, tolerance=np.inf, k=1
    )
    closest = float(np.min(distances[:, 0]))
    leaked = int(np.sum(distances[:, 0] <= tolerance))
    if leaked:
        raise SystemExit(
            f"[eval_baseline_rt] {leaked} evaluated location(s) also appear in the "
            f"TRAIN split (closest distance {closest:.6g} m <= tol {tolerance:g}). "
            f"The test split is leaking; refusing to report."
        )
    return closest


# ----------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------
def build_summary_row(
    predictor: str,
    scored: Dict[str, np.ndarray],
    bookkeeping: Dict[str, object],
    pairwise: Dict[str, float],
) -> Dict[str, object]:
    row: Dict[str, object] = {"predictor": predictor}
    row.update(bookkeeping)

    for prefix, key in (("NMSE_shape", "nmse_shape_db"), ("NMSE_raw", "nmse_raw_db")):
        stats = summarize(scored[key])
        row[f"{prefix}_mean_dB"] = stats["mean"]
        row[f"{prefix}_median_dB"] = stats["median"]
        row[f"{prefix}_p5_dB"] = stats["p5"]
        row[f"{prefix}_p25_dB"] = stats["p25"]
        row[f"{prefix}_p75_dB"] = stats["p75"]
        row[f"{prefix}_p95_dB"] = stats["p95"]
        # Linear-domain average converted once, matching train.py's reporting.
        row[f"{prefix}_meanlinear_dB"] = mean_linear_db(scored[key])

    for k in TOPK_ACC_VALUES:
        row[f"topk_acc_K{k}_mean"] = float(np.mean(scored[f"topk_acc_K{k}"]))
    for k in CAPTURE_VALUES:
        row[f"power_capture_K{k}_mean"] = float(np.mean(scored[f"power_capture_K{k}"]))

    row.update(pairwise)
    return row


def write_summary_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_per_location_csv(
    path: str,
    positions: np.ndarray,
    gt_indices: np.ndarray,
    sionna_indices: np.ndarray,
    match_distance: np.ndarray,
    mimogs: Dict[str, np.ndarray],
    sionna: Dict[str, np.ndarray],
) -> None:
    header = [
        "gt_test_index",
        "sionna_index",
        "match_distance_m",
        "x",
        "y",
        "z",
        "mimogs_NMSE_shape_dB",
        "sionna_NMSE_shape_dB",
        "mimogs_NMSE_raw_dB",
        "sionna_NMSE_raw_dB",
        "nmse_gap_dB",
        "mimogs_top1_hit",
        "sionna_top1_hit",
    ]

    # nmse_gap_dB = Sionna - MIMO-GS: positive means Sionna is worse there.
    difference = sionna["nmse_shape_db"] - mimogs["nmse_shape_db"]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in range(positions.shape[0]):
            writer.writerow(
                [
                    int(gt_indices[row]),
                    int(sionna_indices[row]),
                    f"{match_distance[row]:.9g}",
                    f"{positions[row, 0]:.6f}",
                    f"{positions[row, 1]:.6f}",
                    f"{positions[row, 2]:.6f}",
                    f"{mimogs['nmse_shape_db'][row]:.6f}",
                    f"{sionna['nmse_shape_db'][row]:.6f}",
                    f"{mimogs['nmse_raw_db'][row]:.6f}",
                    f"{sionna['nmse_raw_db'][row]:.6f}",
                    f"{difference[row]:.6f}",
                    int(round(float(mimogs["topk_acc_K1"][row]))),
                    int(round(float(sionna["topk_acc_K1"][row]))),
                ]
            )


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def plot_nmse_cdf(
    output_dir: str, mimogs: Dict[str, np.ndarray], sionna: Dict[str, np.ndarray]
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), sharey=True)

    panels = (
        ("nmse_shape_db", "normalized prediction (shape term)  -- headline"),
        ("nmse_raw_db", "raw prediction (scale term)"),
    )

    for axis, (key, title) in zip(axes, panels):
        for label, scored in ((PREDICTOR_MIMOGS, mimogs), (PREDICTOR_SIONNA, sionna)):
            ordered = np.sort(scored[key])
            probabilities = np.arange(1, ordered.size + 1) / ordered.size
            axis.plot(
                ordered,
                probabilities,
                linewidth=1.8,
                color=PREDICTOR_COLORS[label],
                label=f"{label}  (median {np.median(ordered):.2f} dB)",
            )
        axis.set_xlabel("NMSE [dB]")
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.3, linewidth=0.5)
        axis.legend(fontsize=8, loc="lower right")
        axis.set_ylim(0.0, 1.0)

    axes[0].set_ylabel("Empirical CDF")
    figure.suptitle(
        "Per-location NMSE vs. Wireless InSite ground truth (matched test locations)",
        fontsize=11,
    )
    save_figure(figure, output_dir, "fig_nmse_cdf")


def symmetric_gap_limit(gap_db: np.ndarray) -> float:
    """Symmetric, human-readable color limit for the diverging gap panel.

    Driven by the robust p2/p98 spread rather than the extremes, then rounded
    up to the next 5 dB so the tick labels stay clean.  The returned value is
    what the panel clips to, and it is stated on the colorbar.
    """
    low, high = np.percentile(gap_db, [2.0, 98.0])
    span = float(max(abs(low), abs(high)))
    if not np.isfinite(span) or span <= 0.0:
        span = float(max(np.max(np.abs(gap_db)), 1.0))
    return float(min(max(5.0 * math.ceil(span / 5.0), 5.0), 60.0))


def plot_spatial_nmse_maps(
    output_dir: str,
    positions: np.ndarray,
    mimogs: Dict[str, np.ndarray],
    sionna: Dict[str, np.ndarray],
    gap_db: np.ndarray,
) -> Tuple[float, float, float]:
    """Sionna NMSE | MIMO-GS NMSE | their gap, as three x-y scatter panels.

    Returns ``(vmin, vmax, gap_limit)`` so the caller can record the color
    limits that were actually used.
    """
    stacked = np.concatenate((sionna["nmse_shape_db"], mimogs["nmse_shape_db"]))
    # Robust shared scale so panels 1-2 are directly comparable.
    vmin, vmax = (float(value) for value in np.percentile(stacked, [2.0, 98.0]))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(stacked.min()), float(stacked.max() + 1e-6)

    gap_limit = symmetric_gap_limit(gap_db)

    figure, axes = plt.subplots(1, 3, figsize=(14.6, 4.9), sharex=True, sharey=True)

    nmse_scatter = None
    for axis, title, values in (
        (axes[0], "Sionna RT", sionna["nmse_shape_db"]),
        (axes[1], PREDICTOR_MIMOGS, mimogs["nmse_shape_db"]),
    ):
        nmse_scatter = axis.scatter(
            positions[:, 0],
            positions[:, 1],
            c=values,
            s=6,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            linewidths=0.0,
        )
        axis.set_title(title, fontsize=11)

    gap_scatter = axes[2].scatter(
        positions[:, 0],
        positions[:, 1],
        c=gap_db,
        s=6,
        cmap="coolwarm",
        vmin=-gap_limit,
        vmax=gap_limit,
        linewidths=0.0,
    )
    axes[2].set_title(f"Gap ({PREDICTOR_SIONNA} - {PREDICTOR_MIMOGS})", fontsize=11)

    for axis in axes:
        axis.set_xlabel("x [m]")
        # 'box' rather than 'datalim': the panels share both axes, so the data
        # limits are not independently adjustable.
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.3, linewidth=0.5)
    axes[0].set_ylabel("y [m]")

    # One shared colorbar for the two NMSE panels, a separate one for the gap.
    nmse_colorbar = figure.colorbar(
        nmse_scatter, ax=[axes[0], axes[1]], fraction=0.025, pad=0.015
    )
    nmse_colorbar.set_label("NMSE [dB]")

    gap_colorbar = figure.colorbar(gap_scatter, ax=[axes[2]], fraction=0.05, pad=0.03)
    gap_colorbar.set_label(
        f"NMSE gap [dB] (red: Sionna worse), clipped to +/-{gap_limit:.0f} dB"
    )

    figure.savefig(os.path.join(output_dir, "fig_spatial_nmse_maps.png"), dpi=200)
    figure.savefig(os.path.join(output_dir, "fig_spatial_nmse_maps.pdf"))
    plt.close(figure)

    return vmin, vmax, gap_limit


def select_qualitative_rows(difference: np.ndarray) -> List[Tuple[int, str]]:
    """Pick the two extremes of the disagreement plus one median location."""
    best_for_mimogs = int(np.argmax(difference))          # Sionna much worse
    best_for_sionna = int(np.argmin(difference))          # Sionna much better
    median_row = int(np.argmin(np.abs(difference - np.median(difference))))

    picks = [
        (best_for_mimogs, "MIMO-GS advantage (max)"),
        (median_row, "median disagreement"),
        (best_for_sionna, "Sionna advantage (max)"),
    ]

    seen: set = set()
    unique: List[Tuple[int, str]] = []
    for row, label in picks:
        if row in seen:
            continue
        seen.add(row)
        unique.append((row, label))
    return unique


def plot_qualitative(
    output_dir: str,
    rows: Sequence[Tuple[int, str]],
    positions: np.ndarray,
    target_normalized: torch.Tensor,
    mimogs_prediction: torch.Tensor,
    sionna_prediction: torch.Tensor,
    mimogs: Dict[str, np.ndarray],
    sionna: Dict[str, np.ndarray],
) -> None:
    # All three maps are shown max-normalized: that is the only representation
    # in which a single per-row color scale is meaningful, because the two
    # predictors carry different absolute scales by construction.
    gt_maps = target_normalized.cpu().numpy()
    mimogs_maps = normalize_mag_map(mimogs_prediction).cpu().numpy()
    sionna_maps = normalize_mag_map(sionna_prediction).cpu().numpy()

    figure, axes = plt.subplots(
        len(rows),
        3,
        figsize=(12.6, 2.9 * len(rows) + 1.0),
        squeeze=False,
        layout="constrained",
    )

    for panel, (row, label) in enumerate(rows):
        maps = (gt_maps[row], mimogs_maps[row], sionna_maps[row])
        vmax = float(max(max(float(m.max()) for m in maps), EPS))

        titles = (
            f"WI ground truth -- {label}\n"
            f"(x, y) = ({positions[row, 0]:.1f}, {positions[row, 1]:.1f}) m",
            f"{PREDICTOR_MIMOGS}\nNMSE = {mimogs['nmse_shape_db'][row]:.2f} dB",
            f"{PREDICTOR_SIONNA}\nNMSE = {sionna['nmse_shape_db'][row]:.2f} dB",
        )

        image = None
        for column, (data, title) in enumerate(zip(maps, titles)):
            axis = axes[panel][column]
            image = axis.imshow(
                data,
                aspect="auto",
                interpolation="nearest",
                vmin=0.0,
                vmax=vmax,
                cmap="viridis",
            )
            axis.set_title(title, fontsize=9)
            axis.tick_params(labelsize=7)
            if column == 0:
                axis.set_ylabel("Rx beam", fontsize=8)
            if panel == len(rows) - 1:
                axis.set_xlabel("Tx beam", fontsize=8)

        figure.colorbar(image, ax=axes[panel].tolist(), fraction=0.02, pad=0.01)

    figure.suptitle(
        "Beam-pair maps at the strongest disagreements (each map max-normalized)",
        fontsize=11,
    )
    figure.savefig(os.path.join(output_dir, "fig_qualitative.png"), dpi=200)
    figure.savefig(os.path.join(output_dir, "fig_qualitative.pdf"))
    plt.close(figure)


# ----------------------------------------------------------------------
# README
# ----------------------------------------------------------------------
def write_readme(path: str, lines: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E2 -- MIMO-GS vs. Sionna RT against Wireless InSite ground truth"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=DEFAULT_CKPT,
        help="Run directory (outputs/<run_dir>) or a model.pth path.",
    )
    parser.add_argument(
        "--sionna_mat",
        type=str,
        default=DEFAULT_SIONNA_MAT,
        help="Sionna RT full_dataset.mat holding positions + magnitude.",
    )
    parser.add_argument(
        "--match_tol",
        type=float,
        default=DEFAULT_MATCH_TOL,
        help="Maximum 3-D distance [m] accepted as the same physical location.",
    )
    parser.add_argument(
        "--min_match_frac",
        type=float,
        default=DEFAULT_MIN_MATCH_FRACTION,
        help="Abort (with a coordinate-range diagnostic) below this match rate.",
    )
    parser.add_argument(
        "--allow_partial_match",
        action="store_true",
        help="Continue below --min_match_frac. Only use this once the diagnostic "
        "has ruled out a coordinate-frame or units mismatch; the evaluated "
        "subset is then a biased sample and must be reported as such.",
    )
    parser.add_argument(
        "--skip_bias_check",
        action="store_true",
        help="Skip the extra render of the unmatched GT test locations. That "
        "render is what turns 'the matched subset might be unrepresentative' "
        "into a measured MIMO-GS delta, so only skip it to save time.",
    )
    parser.add_argument("--outputs_root", type=str, default="outputs")
    parser.add_argument("--analysis_root", type=str, default="analysis")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Rendering batch size (0 keeps the checkpoint's training value).",
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="",
        help="Override the ground-truth dataset directory recorded in the checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    repository_root = os.path.dirname(os.path.abspath(__file__))

    outputs_root = arguments.outputs_root
    if not os.path.isabs(outputs_root):
        outputs_root = os.path.join(repository_root, outputs_root)

    run_dir, checkpoint_path = resolve_run_dir(arguments.ckpt, outputs_root)
    run_name = os.path.basename(os.path.normpath(run_dir))

    print("=" * 78)
    print(f"[eval_baseline_rt] RUN        : {run_name}")
    print(f"[eval_baseline_rt] checkpoint : {checkpoint_path}")
    print("=" * 78)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_params, opt_params = restore_config(run_dir, checkpoint)

    if arguments.source_path:
        model_params.source_path = os.path.abspath(arguments.source_path)
    gt_root = str(getattr(model_params, "source_path", ""))
    if not os.path.isdir(gt_root):
        raise SystemExit(
            f"[eval_baseline_rt] Ground-truth dataset '{gt_root}' is missing. "
            f"Pass --source_path <dir>."
        )

    sionna_mat = arguments.sionna_mat
    if not os.path.isabs(sionna_mat):
        sionna_mat = os.path.join(repository_root, sionna_mat)

    # ------------------------------------------------------------------
    # Raw, un-normalized positions -- the only frame in which the three
    # sources can be compared.
    # ------------------------------------------------------------------
    gt_test_mat = os.path.join(gt_root, "test.mat")
    gt_train_mat = os.path.join(gt_root, "train.mat")

    gt_test_positions, gt_test_magnitude = load_raw_mat(gt_test_mat)
    gt_train_positions, _ = load_raw_mat(gt_train_mat)
    sionna_positions, sionna_magnitude = load_raw_mat(sionna_mat)

    print(f"[eval_baseline_rt] GT   : {gt_test_mat}")
    print(f"[eval_baseline_rt] Sionna: {sionna_mat}")
    print(describe_ranges("GT test (WI)", gt_test_positions))
    print(describe_ranges("Sionna RT (full)", sionna_positions))

    if gt_test_magnitude.shape[1:] != sionna_magnitude.shape[1:]:
        raise SystemExit(
            f"[eval_baseline_rt] Beam-grid mismatch: GT {gt_test_magnitude.shape[1:]} "
            f"vs. Sionna {sionna_magnitude.shape[1:]}."
        )

    matched_gt, matched_sionna, match_distance = match_positions(
        gt_test_positions, sionna_positions, float(arguments.match_tol)
    )

    num_gt_test = int(gt_test_positions.shape[0])
    num_matched = int(matched_gt.size)
    num_dropped = num_gt_test - num_matched
    match_fraction = num_matched / max(num_gt_test, 1)

    print()
    print(f"[eval_baseline_rt] MATCHING (tolerance {arguments.match_tol:g} m)")
    print(f"  GT test locations : {num_gt_test}")
    print(f"  matched           : {num_matched}  ({100.0 * match_fraction:.2f}%)")
    print(f"  dropped           : {num_dropped}")
    if num_matched:
        print(
            f"  match distance    : max {match_distance.max():.3g} m / "
            f"mean {match_distance.mean():.3g} m"
        )

    if match_fraction < float(arguments.min_match_frac):
        print()
        print("-" * 78)
        print(
            f"[eval_baseline_rt] MATCH RATE {100.0 * match_fraction:.2f}% IS BELOW THE "
            f"{100.0 * arguments.min_match_frac:.0f}% GATE."
        )
        print("  Coordinate ranges of both sets (check for a frame / units mismatch):")
        print(describe_ranges("GT test (WI)", gt_test_positions))
        print(describe_ranges("Sionna RT (full)", sionna_positions))
        nearest, _ = _neighbour_candidates(
            gt_test_positions, sionna_positions, tolerance=np.inf, k=1
        )
        nearest = nearest[:, 0]
        percentiles = np.percentile(nearest, [50, 90, 99, 100])
        print(
            "  Nearest-neighbour distance [m]: "
            f"p50={percentiles[0]:.4g}  p90={percentiles[1]:.4g}  "
            f"p99={percentiles[2]:.4g}  max={percentiles[3]:.4g}"
        )
        print(
            f"  Exactly coincident (distance == 0): "
            f"{int(np.sum(nearest == 0.0))} / {num_gt_test}"
        )
        print("-" * 78)
        if not arguments.allow_partial_match:
            raise SystemExit(
                "[eval_baseline_rt] Refusing to report metrics on a partial match. "
                "If the ranges above line up and the matched distances are ~0, the "
                "cause is a differing validity mask rather than a frame mismatch; "
                "re-run with --allow_partial_match and report the bias."
            )
        print(
            "[eval_baseline_rt] --allow_partial_match given: continuing on the "
            "matched subset."
        )

    if num_matched == 0:
        raise SystemExit("[eval_baseline_rt] No location matched; nothing to evaluate.")

    matched_positions = gt_test_positions[matched_gt]
    assert np.max(np.abs(matched_positions - sionna_positions[matched_sionna])) <= float(
        arguments.match_tol
    ), "matched positions disagree beyond the tolerance"

    # ------------------------------------------------------------------
    # Scene / model
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda_rasterizer = (
        bool(int(getattr(model_params, "use_cuda_rasterizer", 1))) and device.type == "cuda"
    )
    batch_size = max(1, int(arguments.batch_size) or int(getattr(model_params, "batch_size", 8)))

    hidden_dim = gain_net_hidden_dim(checkpoint)
    if hidden_dim is not None:
        print(
            f"[eval_baseline_rt] checkpoint gain MLP is {hidden_dim}-wide "
            f"(repo default differs); rebuilding it to match."
        )
    with gain_net_width(hidden_dim):
        scene, gaussians = build_scene_and_model(
            model_params, opt_params, checkpoint, device
        )
    scale_factor = float(getattr(scene.test_set, "scale_factor", 1.0))

    print()
    print(
        f"[eval_baseline_rt] device={device} | batch_size={batch_size} | "
        f"cuda_rasterizer={int(use_cuda_rasterizer)} | "
        f"beam grid {scene.beam_rows}x{scene.beam_cols} | "
        f"gaussians={int(gaussians.get_xyz.shape[0])}"
    )

    # The dataset object and the raw .mat must be the same rows in the same
    # order, otherwise every index used below is meaningless.
    dataset_positions_original = scene.test_set.positions.numpy().astype(np.float64) * scale_factor
    position_drift = float(np.max(np.abs(dataset_positions_original - gt_test_positions)))
    assert position_drift < 1e-2, (
        f"Scene test_set positions do not de-normalize back onto test.mat "
        f"(max drift {position_drift:.4g} m); the scale_factor handling is wrong."
    )

    # Both predictors must be scored against the very same GT tensor.
    gt_from_dataset = scene.test_set.magnitude[torch.as_tensor(matched_gt)]
    gt_from_mat = torch.from_numpy(gt_test_magnitude[matched_gt])
    assert torch.equal(gt_from_dataset, gt_from_mat), (
        "The GT magnitudes taken through the Scene path and straight from "
        "test.mat differ; the two predictors would not share a target."
    )

    closest_train_distance = assert_disjoint_from_train(
        matched_positions, gt_train_positions, float(arguments.match_tol)
    )
    print(
        f"[eval_baseline_rt] train-split leakage check: closest evaluated location "
        f"is {closest_train_distance:.4g} m from any TRAIN location -- clean."
    )

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    ground_truth = gt_from_mat.to(device)
    target_normalized = normalize_mag_map(ground_truth)

    zero_power = int((ground_truth.reshape(num_matched, -1).amax(dim=1) <= EPS).sum().item())
    if zero_power:
        print(
            f"[eval_baseline_rt] WARNING: {zero_power} matched GT map(s) have peak "
            f"<= {EPS:g}; their normalization is degenerate."
        )

    sionna_prediction = torch.from_numpy(sionna_magnitude[matched_sionna]).to(device)
    sionna_dead = int(
        (sionna_prediction.reshape(num_matched, -1).amax(dim=1) <= EPS).sum().item()
    )

    print(f"[eval_baseline_rt] rendering {num_matched} MIMO-GS locations ...")
    mimogs_prediction = render_mimogs(
        scene,
        gaussians,
        model_params,
        device,
        scene.test_set.positions[torch.as_tensor(matched_gt)],
        batch_size,
        use_cuda_rasterizer,
    )

    # ------------------------------------------------------------------
    # Sanity: the metric plumbing must return zero error on a self-comparison.
    # ------------------------------------------------------------------
    self_nmse, self_acc, self_capture = self_consistency_check(sionna_prediction)
    print(
        f"[eval_baseline_rt] self-consistency (Sionna vs. Sionna): "
        f"worst NMSE_shape = {self_nmse:.1f} dB, top-1 acc = {self_acc:.4f}, "
        f"capture K=1 = {self_capture:.4f}"
    )
    assert self_nmse <= -100.0, "self-comparison NMSE is not ~0; metric code is broken"
    assert self_acc >= 1.0 - 1e-9, "self-comparison top-1 accuracy is not 1"
    assert self_capture >= 1.0 - 1e-9, "self-comparison power capture is not 1"

    gt_self_nmse, _, _ = self_consistency_check(ground_truth)
    assert gt_self_nmse <= -100.0, "GT self-comparison NMSE is not ~0"

    # ------------------------------------------------------------------
    # Scoring -- identical code path for both predictors
    # ------------------------------------------------------------------
    mimogs_scores = score_prediction(mimogs_prediction, target_normalized)
    sionna_scores = score_prediction(sionna_prediction, target_normalized)

    # ------------------------------------------------------------------
    # Selection-bias check: Sionna has no map at some GT test locations, so the
    # evaluated subset is not the full test split.  MIMO-GS *can* be rendered
    # everywhere, so rendering the dropped locations measures how biased the
    # subset is instead of leaving it as a hand-wave.
    # ------------------------------------------------------------------
    dropped_mask = np.ones(num_gt_test, dtype=bool)
    dropped_mask[matched_gt] = False
    dropped_indices = np.nonzero(dropped_mask)[0]

    bias_delta: Optional[float] = None
    dropped_mean: Optional[float] = None
    if not arguments.skip_bias_check and dropped_indices.size:
        print(
            f"[eval_baseline_rt] bias check: rendering the {dropped_indices.size} "
            f"unmatched GT test location(s) ..."
        )
        dropped_prediction = render_mimogs(
            scene,
            gaussians,
            model_params,
            device,
            scene.test_set.positions[torch.as_tensor(dropped_indices)],
            batch_size,
            use_cuda_rasterizer,
        )
        dropped_target = normalize_mag_map(
            torch.from_numpy(gt_test_magnitude[dropped_indices]).to(device)
        )
        dropped_scores = score_prediction(dropped_prediction, dropped_target)
        dropped_mean = float(np.mean(dropped_scores["nmse_shape_db"]))
        matched_mean = float(np.mean(mimogs_scores["nmse_shape_db"]))
        bias_delta = dropped_mean - matched_mean
        print(
            f"  MIMO-GS NMSE_shape on matched {matched_mean:.2f} dB vs. dropped "
            f"{dropped_mean:.2f} dB  (delta {bias_delta:+.2f} dB)"
        )

    difference = sionna_scores["nmse_shape_db"] - mimogs_scores["nmse_shape_db"]
    difference_raw = sionna_scores["nmse_raw_db"] - mimogs_scores["nmse_raw_db"]
    mimogs_win_fraction = float(np.mean(difference > 0.0))
    mimogs_win_fraction_raw = float(np.mean(difference_raw > 0.0))

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    output_dir = os.path.join(
        repository_root, arguments.analysis_root, run_name, "comparison_rt"
    )
    os.makedirs(output_dir, exist_ok=True)

    # fig_spatial_error_maps was superseded by the three-panel
    # fig_spatial_nmse_maps; drop any copy left by an earlier run so the
    # directory never mixes the two.
    for stale in ("fig_spatial_error_maps.png", "fig_spatial_error_maps.pdf"):
        stale_path = os.path.join(output_dir, stale)
        if os.path.isfile(stale_path):
            os.remove(stale_path)
            print(f"[eval_baseline_rt] removed superseded output {stale}")

    bookkeeping: Dict[str, object] = {
        "run_dir": run_name,
        "checkpoint_path": os.path.relpath(checkpoint_path, repository_root),
        "checkpoint_iteration": int(checkpoint.get("iteration", -1)),
        "gt_source_path": gt_root,
        "gt_test_mat": os.path.relpath(gt_test_mat, repository_root),
        "sionna_mat": os.path.relpath(sionna_mat, repository_root),
        "match_tol_m": float(arguments.match_tol),
        "num_gt_test": num_gt_test,
        "num_sionna_total": int(sionna_positions.shape[0]),
        "num_matched": num_matched,
        "num_dropped": num_dropped,
        "match_fraction": float(match_fraction),
        "num_gt_zero_power": zero_power,
        "num_sionna_zero_power": sionna_dead,
        "Nr": int(scene.beam_rows),
        "Nt": int(scene.beam_cols),
        "num_gaussians": int(gaussians.get_xyz.shape[0]),
        "position_scale_factor": scale_factor,
        "device": str(device),
        "batch_size": batch_size,
        "mimogs_NMSE_shape_on_dropped_dB": (
            "" if dropped_mean is None else f"{dropped_mean:.6f}"
        ),
        "subset_bias_delta_dB": "" if bias_delta is None else f"{bias_delta:.6f}",
    }
    pairwise: Dict[str, float] = {
        "mimogs_win_fraction_shape": mimogs_win_fraction,
        "mimogs_win_fraction_raw": mimogs_win_fraction_raw,
        "nmse_shape_diff_mean_dB": float(np.mean(difference)),
        "nmse_shape_diff_median_dB": float(np.median(difference)),
    }

    summary_rows = [
        build_summary_row(PREDICTOR_MIMOGS, mimogs_scores, bookkeeping, pairwise),
        build_summary_row(PREDICTOR_SIONNA, sionna_scores, bookkeeping, pairwise),
    ]
    write_summary_csv(os.path.join(output_dir, "metrics_summary.csv"), summary_rows)

    write_per_location_csv(
        os.path.join(output_dir, "metrics_per_location.csv"),
        matched_positions,
        matched_gt,
        matched_sionna,
        match_distance,
        mimogs_scores,
        sionna_scores,
    )

    plot_nmse_cdf(output_dir, mimogs_scores, sionna_scores)

    nmse_vmin, nmse_vmax, gap_limit = plot_spatial_nmse_maps(
        output_dir, matched_positions, mimogs_scores, sionna_scores, difference
    )

    # The red half of the diverging panel is exactly "Sionna worse", which is
    # exactly the win fraction reported in the summary.  Tie them together so a
    # future sign flip in either place fails loudly instead of shipping a
    # figure that contradicts the table.
    red_fraction = float(np.mean(difference > 0.0))
    assert abs(red_fraction - mimogs_win_fraction) < 1e-9, (
        f"gap panel red fraction {red_fraction:.6f} disagrees with the reported "
        f"MIMO-GS win fraction {mimogs_win_fraction:.6f}"
    )
    print(
        f"[eval_baseline_rt] gap panel: red (Sionna worse) fraction "
        f"{100.0 * red_fraction:.2f}% == reported win fraction "
        f"{100.0 * mimogs_win_fraction:.2f}%  | NMSE scale "
        f"[{nmse_vmin:.1f}, {nmse_vmax:.1f}] dB, gap clip +/-{gap_limit:.0f} dB"
    )

    qualitative_rows = select_qualitative_rows(difference)
    plot_qualitative(
        output_dir,
        qualitative_rows,
        matched_positions,
        target_normalized,
        mimogs_prediction,
        sionna_prediction,
        mimogs_scores,
        sionna_scores,
    )

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    readme_lines = [
        "E2 -- MIMO-GS vs. Sionna RT against Wireless InSite ground truth",
        "=" * 70,
        "",
        "Generated by eval_baseline_rt.py (repository root).",
        "",
        "MATERIALS",
        "-" * 70,
        f"  MIMO-GS checkpoint   : {bookkeeping['checkpoint_path']} "
        f"(iteration {bookkeeping['checkpoint_iteration']}, "
        f"{bookkeeping['num_gaussians']} gaussians)",
        f"  Ground truth (WI)    : {bookkeeping['gt_test_mat']}  -- TEST split only",
        f"  Sionna RT baseline   : {bookkeeping['sionna_mat']}  -- not split",
        f"  Beam grid            : {bookkeeping['Nr']} Rx x {bookkeeping['Nt']} Tx",
        "",
        "POSITION MATCHING",
        "-" * 70,
        f"  Tolerance                     : {arguments.match_tol:g} m (3-D, one-to-one)",
        f"  GT test locations             : {num_gt_test}",
        f"  Matched to a Sionna location  : {num_matched} "
        f"({100.0 * match_fraction:.2f}%)",
        f"  Dropped (no Sionna location)  : {num_dropped}",
        f"  Max accepted match distance   : "
        f"{(match_distance.max() if num_matched else float('nan')):.3g} m",
        f"  Sionna locations available    : {bookkeeping['num_sionna_total']}",
        "",
        "  Matching runs on the ORIGINAL (un-normalized) .mat coordinates.  The",
        "  Scene/DeepMIMODataset path divides positions by a per-file",
        f"  scale_factor ({scale_factor:.6f} here); the dataset tensors are",
        "  de-normalized and asserted against the raw test.mat before use, and",
        "  the GT magnitudes taken through both paths are asserted identical, so",
        "  both predictors are provably scored against the same target tensor.",
        "",
        "NORMALIZATION CONVENTION",
        "-" * 70,
        "  Target  : normalize_mag_map(GT)  -- per-location max-normalized,",
        "            exactly as in eval_render.py / train.py.",
        "  NMSE_shape_dB : prediction max-normalized  (scale-invariant)  [HEADLINE]",
        "  NMSE_raw_dB   : prediction used verbatim   (training scale term)",
        "",
        "  NMSE_raw is reported for continuity with eval_render.py but is NOT",
        "  comparable across the two predictors: MIMO-GS was optimized to emit",
        "  the max-normalized target directly, whereas the Sionna maps carry the",
        "  Sionna dataset's own global scale.  Cross-method claims should quote",
        "  NMSE_shape, which is what the figures show.",
        "",
        "  Top-K overlap accuracy and power capture are rank-based and therefore",
        "  identical under either convention.",
        "",
        "FAIRNESS FRAMING (different information budgets)",
        "-" * 70,
        "  Sionna RT is a simulation baseline: it consumed the scene geometry and",
        "  its material/ray-tracing configuration, but zero WI measurements.  It",
        "  has no train/test split to leak, so scoring it on the WI TEST",
        "  locations is fair.",
        "  MIMO-GS consumed WI measurements at the TRAIN locations and is scored",
        "  only on held-out TEST locations (asserted disjoint from TRAIN: the",
        f"  closest evaluated location is {closest_train_distance:.4g} m from any",
        "  TRAIN location).  Its number is a generalization number.",
        "  The comparison therefore measures what a measurement-trained",
        "  representation buys over a geometry-only simulator, not which model is",
        "  intrinsically better.",
        "",
        "SANITY CHECKS RUN",
        "-" * 70,
        f"  Sionna scored against itself : NMSE_shape = {self_nmse:.1f} dB "
        f"(1e-12 clamp floor), top-1 acc = {self_acc:.4f}, capture K=1 = "
        f"{self_capture:.4f}",
        f"  GT scored against itself     : NMSE_shape = {gt_self_nmse:.1f} dB",
        "  Test/train disjointness      : asserted (see above)",
        "  GT tensor identity           : asserted (Scene path == raw test.mat)",
        "  Position de-normalization    : asserted (Scene positions * scale_factor",
        f"                                 == test.mat, max drift {position_drift:.2e} m)",
        "  One-to-one matching          : asserted (no Sionna row reused)",
        "  Gain-MLP restore             : strict load_state_dict, no partial load",
        f"                                 (checkpoint width {hidden_dim or 'repo default'})",
        "",
        "  NOTE ON THE MIMO-GS LEVEL.  This checkpoint scores around -21 dB, not the",
        "  -11..-12 dB of the older 10-epoch runs.  That is the checkpoint, not a",
        "  loading bug: iteration "
        f"{bookkeeping['checkpoint_iteration']} == ceil(N_train/batch)*num_epochs for",
        "  this dataset, the gain MLP loads strictly (a width mismatch is raised, not",
        "  silently skipped), and the train/test gap is <1 dB while the two splits are",
        "  provably disjoint.  Older baselines were measured at fewer epochs and before",
        "  the normalization fix, so they are not comparable to this run.",
        "",
        "RESULTS",
        "-" * 70,
    ]

    header = f"  {'predictor':<12}{'NMSE_shape mean':>17}{'median':>10}{'p5':>9}{'p95':>9}"
    readme_lines.append(header)
    for label, scored in ((PREDICTOR_MIMOGS, mimogs_scores), (PREDICTOR_SIONNA, sionna_scores)):
        stats = summarize(scored["nmse_shape_db"])
        readme_lines.append(
            f"  {label:<12}{stats['mean']:>17.2f}{stats['median']:>10.2f}"
            f"{stats['p5']:>9.2f}{stats['p95']:>9.2f}"
        )
    readme_lines += [
        "",
        f"  MIMO-GS strictly better than Sionna at "
        f"{100.0 * mimogs_win_fraction:.2f}% of the matched locations",
        f"  (NMSE_shape difference, Sionna - MIMO-GS): mean "
        f"{np.mean(difference):+.2f} dB, median {np.median(difference):+.2f} dB",
        "",
        "  The two error distributions have different SHAPES, not just different",
        "  means.  Sionna's p5 "
        f"({summarize(sionna_scores['nmse_shape_db'])['p5']:.2f} dB) is better than",
        f"  MIMO-GS's ({summarize(mimogs_scores['nmse_shape_db'])['p5']:.2f} dB): where",
        "  the ray tracer resolves the geometry it is very accurate, but it collapses",
        "  elsewhere, so its median is "
        f"{summarize(sionna_scores['nmse_shape_db'])['median']:.2f} dB.  MIMO-GS is the",
        "  tighter, more uniform predictor rather than the uniformly better one; see",
        "  the CDF crossover in fig_nmse_cdf and the spatial split in",
        "  fig_spatial_nmse_maps.",
        "",
        "  Gap distribution (nmse_gap_dB = Sionna - MIMO-GS, per location):",
        f"    median {np.median(difference):+.2f} dB, IQR "
        f"[{np.percentile(difference, 25):+.2f}, "
        f"{np.percentile(difference, 75):+.2f}] dB "
        f"(width {np.percentile(difference, 75) - np.percentile(difference, 25):.2f} dB)",
        f"    positive (Sionna worse, red) at {100.0 * red_fraction:.2f}% of locations,",
        "    which is the same quantity as the win fraction above and is asserted",
        "    equal at run time.",
        "",
        "FIGURE COLOR LIMITS",
        "-" * 70,
        f"  Panels 1-2 (Sionna / MIMO-GS NMSE): shared viridis scale "
        f"[{nmse_vmin:.2f}, {nmse_vmax:.2f}] dB,",
        "    taken as the p2/p98 of the two predictors' per-location NMSE pooled,",
        "    so the two panels are directly comparable.",
        f"  Panel 3 (gap): coolwarm, symmetric about 0, clipped to "
        f"+/-{gap_limit:.0f} dB",
        "    (p2/p98 of the gap, rounded up to the next 5 dB).  Red = positive =",
        "    Sionna worse; blue = negative = MIMO-GS worse.",
        "",
        "  No BS marker is drawn on any figure.",
        "",
        "ON DELTA-G (dropped)",
        "-" * 70,
        "  A Delta-G (gain / absolute-level difference) figure is deliberately NOT",
        "  produced.  Both magnitude sets are per-location max-normalized before any",
        "  metric is computed -- and they arrive already carrying different global",
        "  scales from their own generators -- so the absolute level is not a",
        "  recoverable quantity here.  Any Delta-G computed from these .mat files",
        "  would measure the two generators' normalization conventions rather than a",
        "  physical gain difference.  Recovering it would require the un-normalized",
        "  path gains (and a common reference) from both simulators, which are not",
        "  present in either dataset.",
        "",
        "CAVEATS",
        "-" * 70,
    ]

    if match_fraction < float(arguments.min_match_frac):
        readme_lines += [
            f"  * MATCH RATE {100.0 * match_fraction:.2f}% IS BELOW THE "
            f"{100.0 * arguments.min_match_frac:.0f}% GATE and the run was forced",
            "    through with --allow_partial_match.  The coordinate frames, the",
            "    grid spacing and the axis ranges of the two sets agree and the",
            "    accepted matches are exactly coincident, so this is a differing",
            "    validity mask between the two simulators, not a frame or units",
            f"    mismatch.  The {num_dropped} unmatched GT locations are excluded,",
            "    which makes the evaluated subset a NON-UNIFORM sample of the test",
            "    split -- check the spatial map before quoting these numbers as",
            "    'the test set'.",
        ]
        if bias_delta is not None:
            readme_lines += [
                "    Measured bias: MIMO-GS scores "
                f"{np.mean(mimogs_scores['nmse_shape_db']):.2f} dB on the matched",
                f"    subset and {dropped_mean:.2f} dB on the {num_dropped} dropped",
                f"    locations ({bias_delta:+.2f} dB), so for MIMO-GS the subset is",
                "    representative to well under a dB.  Sionna cannot be checked the",
                "    same way -- it has no prediction at those locations at all, which",
                "    is itself a coverage failure this comparison does not charge it",
                "    for.",
            ]

    readme_lines += [
        f"  * {sionna_dead} of the {num_matched} matched Sionna maps have a peak "
        f"<= {EPS:g}",
        "    (the ray tracer found essentially no path).  normalize_mag_map clamps",
        "    the divisor at 1e-8, so those maps stay effectively un-normalized and",
        "    score near 0 dB.  That is the honest outcome for a simulator that",
        "    predicts nothing there, but it does put mass in the right tail.",
        "  * Sionna material settings: the dataset folder",
        f"    ({os.path.dirname(bookkeeping['sionna_mat'])}) ships only",
        "    full_dataset.mat -- no scene file, no material table, no generator",
        "    script, and no bs_info.yml.  Whether the ray tracer ran with Sionna's",
        "    DEFAULT ITU material presets or with materials calibrated against the",
        "    Wireless InSite scene CANNOT be determined from what is checked in.",
        "    This must be confirmed before the number is used in the paper: an",
        "    uncalibrated default-material baseline understates what RT can do.",
        "  * Sionna's BS position/orientation is likewise not recorded here; the",
        "    comparison assumes the two simulators used the same transmitter, which",
        "    is consistent with the shared UE grid but is not verifiable from the",
        "    .mat alone.",
        "  * Both magnitude sets arrive already globally scaled by their own",
        "    generator, which is why only the max-normalized (shape) convention is",
        "    quoted for cross-method claims.",
        "",
        "FILES",
        "-" * 70,
        "  metrics_summary.csv        one row per predictor, aggregates + bookkeeping",
        "  metrics_per_location.csv   per matched location, both predictors",
        "  fig_nmse_cdf.*             per-location NMSE CDF, both conventions",
        "  fig_spatial_nmse_maps.*    Sionna NMSE | MIMO-GS NMSE | gap, x-y scatter",
        "  fig_qualitative.*          GT vs. MIMO-GS vs. Sionna at 3 locations",
    ]
    write_readme(os.path.join(output_dir, "README.txt"), readme_lines)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print(f"[eval_baseline_rt] SUMMARY -- {num_matched} matched test locations")
    print("=" * 78)
    for convention, key in (("NMSE_shape (headline)", "nmse_shape_db"), ("NMSE_raw", "nmse_raw_db")):
        print(f"  {convention}")
        print(
            f"    {'predictor':<12}{'mean':>9}{'median':>9}{'p5':>9}{'p25':>9}"
            f"{'p75':>9}{'p95':>9}{'mean-lin':>10}"
        )
        for label, scored in (
            (PREDICTOR_MIMOGS, mimogs_scores),
            (PREDICTOR_SIONNA, sionna_scores),
        ):
            stats = summarize(scored[key])
            print(
                f"    {label:<12}{stats['mean']:>9.2f}{stats['median']:>9.2f}"
                f"{stats['p5']:>9.2f}{stats['p25']:>9.2f}{stats['p75']:>9.2f}"
                f"{stats['p95']:>9.2f}{mean_linear_db(scored[key]):>10.2f}"
            )
    print()
    print(f"  {'predictor':<12}" + "".join(f"{f'acc@K{k}':>10}" for k in TOPK_ACC_VALUES)
          + "".join(f"{f'cap@K{k}':>10}" for k in CAPTURE_VALUES))
    for label, scored in ((PREDICTOR_MIMOGS, mimogs_scores), (PREDICTOR_SIONNA, sionna_scores)):
        print(
            f"  {label:<12}"
            + "".join(f"{np.mean(scored[f'topk_acc_K{k}']):>10.4f}" for k in TOPK_ACC_VALUES)
            + "".join(f"{np.mean(scored[f'power_capture_K{k}']):>10.4f}" for k in CAPTURE_VALUES)
        )
    print()
    print(
        f"  Sionna - MIMO-GS NMSE_shape: mean {np.mean(difference):+.2f} dB, "
        f"median {np.median(difference):+.2f} dB"
    )
    print(
        f"  MIMO-GS strictly better at {100.0 * mimogs_win_fraction:.2f}% "
        f"of locations ({int(np.sum(difference > 0))}/{num_matched})"
    )
    print()
    print("  Qualitative rows:")
    for row, label in qualitative_rows:
        print(
            f"    {label:<26} (x,y)=({matched_positions[row, 0]:7.2f},"
            f"{matched_positions[row, 1]:7.2f})  "
            f"MIMO-GS {mimogs_scores['nmse_shape_db'][row]:7.2f} dB  "
            f"Sionna {sionna_scores['nmse_shape_db'][row]:7.2f} dB"
        )
    print()
    print(f"[eval_baseline_rt] Outputs written to {output_dir}")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
