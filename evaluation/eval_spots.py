#!/usr/bin/env python3
"""D3 -- qualitative beam-pair maps at the locations where the methods disagree.

Compares, on the ORIGINAL full test set,

  * MIMO-GS   outputs/density/mimogs/model_100.pth
  * MLP       outputs/density/MLP/model_100.pth
  * Sionna RT the ray-traced maps ``evaluation/eval_baseline_rt.py`` uses
              (``dataset/asu_sionna_16by64_lt/full_dataset.mat``), matched onto
              the test positions in ORIGINAL meters -- skipped gracefully when
              the file is absent, and skipped per location when a given test
              position has no RT counterpart.

Two candidate criteria are evaluated over the whole test set:

  (i)  largest per-location NMSE gap, MLP minus MIMO-GS;
  (ii) largest top-4 beam-set disagreement, |top4(GT) \\ top4(method)| for MLP
       minus the same count for MIMO-GS.

The union of the top-15 under each criterion becomes the candidate gallery.

Zero-argument runnable::

    python eval_spots.py                 # candidate gallery
    python eval_spots.py --spots 12,34,56  # publication figure for three spots

Nothing in the repository is modified.  Metrics, checkpoint loading and figure
conventions come from ``eval_density.py`` (and through it from
``evaluation/eval_render.py``).
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

from eval_density import (
    AXIS_LABEL_FONTSIZE,
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_MIMOGS_DIR,
    DEFAULT_MLP_DIR,
    FIGURE_DPI,
    METHOD_MIMOGS,
    METHOD_MLP,
    METHOD_RT,
    REPO_ROOT,
    TICK_LABELSIZE,
    TestGroundTruth,
    assert_finite_nonnegative,
    load_mimogs,
    load_mlp,
    load_raw_mat,
    match_positions,
    predict_mlp_maps,
    render_mimogs_maps,
    resolve_device,
    score_prediction,
    write_csv,
    write_readme,
)
import eval_render as ER
from utils.loss import normalize_mag_map


METHOD_GT = "Ground truth"
ROW_ORDER: Tuple[str, ...] = (METHOD_GT, METHOD_MIMOGS, METHOD_MLP, METHOD_RT)

DEFAULT_SIONNA_MAT = os.path.join(
    REPO_ROOT, "dataset", "asu_sionna_16by64_lt", "full_dataset.mat"
)
SIONNA_MATCH_TOL = 1e-3

TOP_CANDIDATES_PER_CRITERION = 15
DB_FLOOR = -30.0
SELF_CHECK_TOLERANCE_DB = 0.05

DENSITY_CSV_RELATIVE = os.path.join("eval_density", "density_metrics.csv")


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
def collect_predictions(
    arguments: argparse.Namespace, device: torch.device
) -> Dict[str, object]:
    """Render/collect every method's maps over the full test set."""
    mimogs_path = os.path.join(arguments.mimogs_dir, "model_100.pth")
    mlp_path = os.path.join(arguments.mlp_dir, "model_100.pth")

    probe = torch.load(mimogs_path, map_location="cpu", weights_only=False)
    dataset_dir = os.path.abspath(arguments.dataset or probe["config"]["dataset_path"])
    del probe

    ground_truth = TestGroundTruth(dataset_dir, device)

    loaded_gs = load_mimogs(mimogs_path, device, dataset_dir)
    gs_maps = render_mimogs_maps(loaded_gs, ground_truth.positions_normalized)
    assert_finite_nonnegative(gs_maps, "MIMO-GS model_100")
    gs_scored = ground_truth.score(gs_maps)

    loaded_mlp = load_mlp(mlp_path, device)
    mlp_maps = predict_mlp_maps(loaded_mlp, ground_truth.positions_normalized)
    assert_finite_nonnegative(mlp_maps, "MLP model_100")
    mlp_scored = ground_truth.score(mlp_maps)

    maps: Dict[str, torch.Tensor] = {
        METHOD_GT: ground_truth.magnitude[
            torch.as_tensor(ground_truth.valid_indices, device=device)
        ],
        METHOD_MIMOGS: gs_maps[torch.as_tensor(ground_truth.valid_indices, device=device)],
        METHOD_MLP: mlp_maps[torch.as_tensor(ground_truth.valid_indices, device=device)],
    }
    scored: Dict[str, Dict[str, np.ndarray]] = {
        METHOD_MIMOGS: gs_scored,
        METHOD_MLP: mlp_scored,
    }

    # -- Sionna RT --------------------------------------------------------
    rt_note = ""
    rt_row_available = False
    rt_matched_mask = np.zeros(ground_truth.num_scored, dtype=bool)

    sionna_mat = arguments.sionna_mat or DEFAULT_SIONNA_MAT
    if not os.path.isfile(sionna_mat):
        rt_note = f"Sionna RT skipped: '{sionna_mat}' is not on disk."
    else:
        rt_positions, rt_magnitude = load_raw_mat(sionna_mat)
        gt_indices, rt_indices, _ = match_positions(
            ground_truth.valid_positions_m, rt_positions, SIONNA_MATCH_TOL
        )
        if gt_indices.size == 0:
            rt_note = (
                f"Sionna RT skipped: no test position in '{os.path.basename(sionna_mat)}' "
                f"matched within {SIONNA_MATCH_TOL} m."
            )
        else:
            rt_row_available = True
            rt_matched_mask[gt_indices] = True

            rt_maps_full = torch.zeros_like(maps[METHOD_GT])
            rt_maps_full[torch.as_tensor(gt_indices, device=device)] = torch.as_tensor(
                np.ascontiguousarray(rt_magnitude[rt_indices]),
                dtype=torch.float32,
                device=device,
            )
            maps[METHOD_RT] = rt_maps_full

            rt_scored_subset = score_prediction(
                rt_maps_full[torch.as_tensor(gt_indices, device=device)],
                ground_truth.target_normalized[torch.as_tensor(gt_indices, device=device)],
            )
            rt_scored: Dict[str, np.ndarray] = {}
            for key, values in rt_scored_subset.items():
                full = np.full(ground_truth.num_scored, np.nan, dtype=np.float64)
                full[gt_indices] = values
                rt_scored[key] = full
            scored[METHOD_RT] = rt_scored

            rt_note = (
                f"Sionna RT matched {int(gt_indices.size)} of "
                f"{ground_truth.num_scored} scored test locations "
                f"(tolerance {SIONNA_MATCH_TOL} m) from "
                f"{os.path.relpath(sionna_mat, REPO_ROOT)}."
            )

    return {
        "dataset_dir": dataset_dir,
        "ground_truth": ground_truth,
        "maps": maps,
        "scored": scored,
        "rt_available": rt_row_available,
        "rt_matched_mask": rt_matched_mask,
        "rt_note": rt_note,
        "checkpoints": {
            METHOD_MIMOGS: os.path.relpath(mimogs_path, REPO_ROOT),
            METHOD_MLP: os.path.relpath(mlp_path, REPO_ROOT),
        },
    }


# ---------------------------------------------------------------------------
# Self-check against eval_density
# ---------------------------------------------------------------------------
def read_density_reference(analysis_root: str) -> Optional[float]:
    """MIMO-GS model_100 shape NMSE as recorded by ``eval_density.py``."""
    path = os.path.join(analysis_root, DENSITY_CSV_RELATIVE)
    for row in ER.read_csv_rows(path):
        if row.get("method", "").strip() != METHOD_MIMOGS:
            continue
        if abs((ER._as_float(row.get("fraction")) or 0.0) - 1.0) > 1e-9:
            continue
        return ER._as_float(row.get("nmse_shape_mean_dB"))
    return None


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------
def top4_missing(scored: Dict[str, np.ndarray]) -> np.ndarray:
    """``|top4(GT) \\ top4(method)|`` from eval_render's top-K overlap."""
    return 4.0 * (1.0 - np.asarray(scored["topk_acc_K4"], dtype=np.float64))


def select_candidates(
    scored: Dict[str, Dict[str, np.ndarray]], how_many: int
) -> Dict[str, object]:
    gap_nmse = (
        scored[METHOD_MLP]["nmse_shape_db"] - scored[METHOD_MIMOGS]["nmse_shape_db"]
    )
    gap_top4 = top4_missing(scored[METHOD_MLP]) - top4_missing(scored[METHOD_MIMOGS])

    by_nmse = np.argsort(-gap_nmse, kind="stable")[:how_many]
    by_top4 = np.argsort(-gap_top4, kind="stable")[:how_many]

    union: List[int] = []
    for row in list(by_nmse) + list(by_top4):
        if int(row) not in union:
            union.append(int(row))

    return {
        "gap_nmse": gap_nmse,
        "gap_top4": gap_top4,
        "by_nmse": by_nmse.astype(np.int64),
        "by_top4": by_top4.astype(np.int64),
        "union": np.asarray(sorted(union), dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# Map rendering helpers
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
        return 0.0, 1.0, "Normalized magnitude"
    return DB_FLOOR, 0.0, "Normalized magnitude [dB]"


def rows_for_location(
    row: int, predictions: Dict[str, object]
) -> List[str]:
    """Method rows available at one scored test location."""
    available = [METHOD_GT, METHOD_MIMOGS, METHOD_MLP]
    if predictions["rt_available"] and bool(predictions["rt_matched_mask"][row]):
        available.append(METHOD_RT)
    return available


def location_nmse(row: int, method: str, predictions: Dict[str, object]) -> float:
    if method == METHOD_GT:
        return float("nan")
    scored = predictions["scored"].get(method)
    if scored is None:
        return float("nan")
    return float(scored["nmse_shape_db"][row])


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------
def render_gallery_figure(
    output_dir: str,
    row: int,
    predictions: Dict[str, object],
    scale: str,
) -> None:
    ground_truth: TestGroundTruth = predictions["ground_truth"]
    methods = rows_for_location(row, predictions)
    vmin, vmax, colorbar_label = panel_limits(scale)
    convert = to_linear_panel if scale == "linear" else to_db_panel

    test_index = int(ground_truth.valid_indices[row])
    position = ground_truth.valid_positions_m[row]

    figure, axes = plt.subplots(
        len(methods),
        1,
        figsize=(7.4, 1.35 * len(methods) + 1.05),
        squeeze=False,
        layout="constrained",
    )

    image = None
    for panel, method in enumerate(methods):
        axis = axes[panel][0]
        data = convert(predictions["maps"][method][row])
        image = axis.imshow(
            data, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax,
            cmap="viridis",
        )
        axis.set_ylabel(method, fontsize=AXIS_LABEL_FONTSIZE)
        axis.tick_params(labelsize=TICK_LABELSIZE)
        if panel == len(methods) - 1:
            axis.set_xlabel("Tx beam index", fontsize=AXIS_LABEL_FONTSIZE)
        else:
            axis.set_xticklabels([])

        value = location_nmse(row, method, predictions)
        if np.isfinite(value):
            axis.text(
                0.995,
                0.94,
                f"NMSE {value:.2f} dB",
                transform=axis.transAxes,
                fontsize=8,
                ha="right",
                va="top",
                color="black",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.4),
            )

        # The location caption lives inside the ground-truth panel so it can
        # never collide with the axis labels under a constrained layout.
        if panel == 0:
            axis.text(
                0.005,
                0.94,
                f"test index {test_index}   "
                f"(x, y, z) = ({position[0]:.2f}, {position[1]:.2f}, "
                f"{position[2]:.2f}) m\n"
                f"NMSE gap (MLP - MIMO-GS) = "
                f"{float(predictions['candidates']['gap_nmse'][row]):+.2f} dB   "
                f"top-4 miss gap = "
                f"{float(predictions['candidates']['gap_top4'][row]):+.2f}",
                transform=axis.transAxes,
                fontsize=8,
                ha="left",
                va="top",
                color="black",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.4),
            )

    colorbar = figure.colorbar(
        image, ax=[axis for row_axes in axes for axis in row_axes], fraction=0.03, pad=0.015
    )
    colorbar.set_label(colorbar_label, fontsize=10)
    colorbar.ax.tick_params(labelsize=9)

    target_dir = os.path.join(output_dir, "gallery", scale)
    os.makedirs(target_dir, exist_ok=True)
    figure.savefig(os.path.join(target_dir, f"loc_{test_index}.png"), dpi=FIGURE_DPI)
    figure.savefig(os.path.join(target_dir, f"loc_{test_index}.pdf"))
    plt.close(figure)


# ---------------------------------------------------------------------------
# Final publication figure
# ---------------------------------------------------------------------------
def render_spot_grid(
    output_dir: str,
    rows: Sequence[int],
    predictions: Dict[str, object],
    scale: str,
) -> None:
    ground_truth: TestGroundTruth = predictions["ground_truth"]
    vmin, vmax, colorbar_label = panel_limits(scale)
    convert = to_linear_panel if scale == "linear" else to_db_panel

    methods = [METHOD_GT, METHOD_MIMOGS, METHOD_MLP]
    if predictions["rt_available"] and any(
        bool(predictions["rt_matched_mask"][row]) for row in rows
    ):
        methods.append(METHOD_RT)

    figure, axes = plt.subplots(
        len(methods),
        len(rows),
        figsize=(3.3 * len(rows) + 0.9, 1.30 * len(methods) + 0.85),
        squeeze=False,
        layout="constrained",
    )

    image = None
    for column, row in enumerate(rows):
        test_index = int(ground_truth.valid_indices[row])
        for panel, method in enumerate(methods):
            axis = axes[panel][column]
            has_data = method != METHOD_RT or bool(predictions["rt_matched_mask"][row])
            if has_data:
                image = axis.imshow(
                    convert(predictions["maps"][method][row]),
                    aspect="auto",
                    interpolation="nearest",
                    vmin=vmin,
                    vmax=vmax,
                    cmap="viridis",
                )
            else:
                axis.set_facecolor("0.92")
                axis.text(
                    0.5, 0.5, "no RT match", transform=axis.transAxes,
                    fontsize=8, ha="center", va="center", color="0.35",
                )
                axis.set_xticks([])
                axis.set_yticks([])

            axis.tick_params(labelsize=TICK_LABELSIZE)
            if column == 0:
                axis.set_ylabel(method, fontsize=AXIS_LABEL_FONTSIZE)
            elif has_data:
                axis.set_yticklabels([])
            if panel == len(methods) - 1:
                axis.set_xlabel("Tx beam index", fontsize=AXIS_LABEL_FONTSIZE)
            elif has_data:
                axis.set_xticklabels([])

            if panel == 0:
                axis.text(
                    0.012, 0.92, f"loc {test_index}", transform=axis.transAxes,
                    fontsize=8, ha="left", va="top",
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.4),
                )
            value = location_nmse(row, method, predictions)
            if has_data and np.isfinite(value):
                axis.text(
                    0.988, 0.92, f"{value:.1f} dB", transform=axis.transAxes,
                    fontsize=8, ha="right", va="top",
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.4),
                )

    colorbar = figure.colorbar(
        image, ax=[axis for row_axes in axes for axis in row_axes], fraction=0.02, pad=0.012
    )
    colorbar.set_label(colorbar_label, fontsize=10)
    colorbar.ax.tick_params(labelsize=9)

    os.makedirs(output_dir, exist_ok=True)
    stem = f"fig_qualitative_spots_{scale}"
    figure.savefig(os.path.join(output_dir, f"{stem}.png"), dpi=FIGURE_DPI)
    figure.savefig(os.path.join(output_dir, f"{stem}.pdf"))
    plt.close(figure)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def write_candidate_csv(
    path: str, predictions: Dict[str, object], candidates: Dict[str, object]
) -> None:
    ground_truth: TestGroundTruth = predictions["ground_truth"]
    by_nmse = set(int(v) for v in candidates["by_nmse"])
    by_top4 = set(int(v) for v in candidates["by_top4"])

    header = [
        "test_index",
        "x_m",
        "y_m",
        "z_m",
        "criterion_nmse_gap_dB",
        "criterion_top4_miss_gap",
        "selected_by_nmse_gap",
        "selected_by_top4_gap",
        "nmse_shape_dB_mimogs",
        "nmse_shape_dB_mlp",
        "nmse_shape_dB_sionna_rt",
        "top4_miss_mimogs",
        "top4_miss_mlp",
        "top4_miss_sionna_rt",
        "sionna_rt_matched",
    ]
    rows = []
    rt_scored = predictions["scored"].get(METHOD_RT)
    for row in candidates["union"]:
        row = int(row)
        position = ground_truth.valid_positions_m[row]
        matched = bool(predictions["rt_matched_mask"][row])
        rt_nmse = (
            f"{float(rt_scored['nmse_shape_db'][row]):.6f}"
            if rt_scored is not None and matched
            else ""
        )
        rt_miss = (
            f"{float(top4_missing(rt_scored)[row]):.6f}"
            if rt_scored is not None and matched
            else ""
        )
        rows.append(
            [
                int(ground_truth.valid_indices[row]),
                f"{position[0]:.6f}",
                f"{position[1]:.6f}",
                f"{position[2]:.6f}",
                f"{float(candidates['gap_nmse'][row]):.6f}",
                f"{float(candidates['gap_top4'][row]):.6f}",
                int(row in by_nmse),
                int(row in by_top4),
                f"{float(predictions['scored'][METHOD_MIMOGS]['nmse_shape_db'][row]):.6f}",
                f"{float(predictions['scored'][METHOD_MLP]['nmse_shape_db'][row]):.6f}",
                rt_nmse,
                f"{float(top4_missing(predictions['scored'][METHOD_MIMOGS])[row]):.6f}",
                f"{float(top4_missing(predictions['scored'][METHOD_MLP])[row]):.6f}",
                rt_miss,
                int(matched),
            ]
        )
    write_csv(path, header, rows)


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
            raise SystemExit(f"[eval_spots] --spots value is not an integer: {token!r}") from error
    return values


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="D3 -- qualitative beam-pair maps where MIMO-GS and the MLP disagree"
    )
    parser.add_argument("--mimogs_dir", type=str, default=DEFAULT_MIMOGS_DIR)
    parser.add_argument("--mlp_dir", type=str, default=DEFAULT_MLP_DIR)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--analysis_root", type=str, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument(
        "--sionna_mat",
        type=str,
        default="",
        help=f"Sionna RT full_dataset.mat (default: {os.path.relpath(DEFAULT_SIONNA_MAT, REPO_ROOT)})",
    )
    parser.add_argument(
        "--spots",
        type=str,
        default="",
        help="Final-figure mode: comma-separated test indices, e.g. --spots 12,34,56",
    )
    parser.add_argument(
        "--gallery",
        action="store_true",
        help="Also render the candidate gallery when --spots is given.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    return parser


def main() -> int:
    arguments = build_argument_parser().parse_args()
    device = resolve_device(prefer_cuda=not arguments.cpu)
    warnings: List[str] = []

    print("=" * 100)
    print("[eval_spots] Qualitative beam-pair maps at the largest MLP / MIMO-GS gaps")
    print("=" * 100)
    print(f"[eval_spots] device : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    predictions = collect_predictions(arguments, device)
    ground_truth: TestGroundTruth = predictions["ground_truth"]

    print(f"[eval_spots] dataset: {predictions['dataset_dir']}")
    print(f"[eval_spots] scored test locations: {ground_truth.num_scored} "
          f"(skipped zero-power: {ground_truth.num_skipped_zero_power})")
    print(f"[eval_spots] {predictions['rt_note']}")
    if not predictions["rt_available"]:
        warnings.append(f"WARN {predictions['rt_note']}")

    # -- self-check -------------------------------------------------------
    measured = float(
        np.mean(predictions["scored"][METHOD_MIMOGS]["nmse_shape_db"])
    )
    reference = read_density_reference(arguments.analysis_root)
    print("")
    print("-" * 100)
    print("[eval_spots] SELF-CHECK -- MIMO-GS model_100, full test set")
    print("-" * 100)
    print(f"  this script                     : {measured:.4f} dB")
    if reference is None:
        print(f"  eval_density reference          : not found "
              f"({os.path.join(arguments.analysis_root, DENSITY_CSV_RELATIVE)})")
        warnings.append(
            "WARN self-check skipped: no eval_density density_metrics.csv on disk. "
            "Run 'python eval_density.py' first."
        )
    else:
        delta = abs(measured - float(reference))
        status = "ok" if delta <= SELF_CHECK_TOLERANCE_DB else "MISMATCH"
        print(f"  eval_density reference          : {float(reference):.4f} dB")
        print(f"  delta                           : {delta:.4f} dB "
              f"(tolerance {SELF_CHECK_TOLERANCE_DB:.2f} dB) -> {status}")
        if status != "ok":
            warnings.append(
                f"WARN self-check MISMATCH: {measured:.4f} dB here vs. "
                f"{float(reference):.4f} dB in eval_density (delta {delta:.4f} dB). "
                f"Same loading path, so this indicates a bug in eval_spots.py."
            )

    # -- candidates -------------------------------------------------------
    candidates = select_candidates(predictions["scored"], TOP_CANDIDATES_PER_CRITERION)
    predictions["candidates"] = candidates

    output_dir = os.path.join(arguments.analysis_root, "eval_spots")
    os.makedirs(output_dir, exist_ok=True)

    print("")
    print("-" * 100)
    print(f"[eval_spots] CANDIDATES -- top {TOP_CANDIDATES_PER_CRITERION} per criterion, "
          f"union of {int(candidates['union'].size)} unique locations")
    print("-" * 100)
    header = (
        f"  {'test idx':>9}{'x [m]':>10}{'y [m]':>10}"
        f"{'NMSE gap':>11}{'top-4 gap':>11}"
        f"{'GS [dB]':>10}{'MLP [dB]':>10}{'RT [dB]':>10}  criteria"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    rt_scored = predictions["scored"].get(METHOD_RT)
    by_nmse = set(int(v) for v in candidates["by_nmse"])
    by_top4 = set(int(v) for v in candidates["by_top4"])
    for row in candidates["union"]:
        row = int(row)
        position = ground_truth.valid_positions_m[row]
        matched = bool(predictions["rt_matched_mask"][row])
        rt_text = (
            f"{float(rt_scored['nmse_shape_db'][row]):>10.2f}"
            if rt_scored is not None and matched
            else f"{'n/a':>10}"
        )
        tags = ",".join(
            tag for tag, member in (("nmse", row in by_nmse), ("top4", row in by_top4)) if member
        )
        print(
            f"  {int(ground_truth.valid_indices[row]):>9}"
            f"{position[0]:>10.2f}{position[1]:>10.2f}"
            f"{float(candidates['gap_nmse'][row]):>11.2f}"
            f"{float(candidates['gap_top4'][row]):>11.2f}"
            f"{float(predictions['scored'][METHOD_MIMOGS]['nmse_shape_db'][row]):>10.2f}"
            f"{float(predictions['scored'][METHOD_MLP]['nmse_shape_db'][row]):>10.2f}"
            f"{rt_text}  {tags}"
        )
    print("  " + "-" * (len(header) - 2))

    write_candidate_csv(os.path.join(output_dir, "candidates.csv"), predictions, candidates)

    # -- figures ----------------------------------------------------------
    spot_rows: List[int] = []
    if arguments.spots:
        requested = parse_spot_argument(arguments.spots)
        lookup = {
            int(value): position
            for position, value in enumerate(ground_truth.valid_indices.tolist())
        }
        missing = [value for value in requested if value not in lookup]
        if missing:
            raise SystemExit(
                f"[eval_spots] --spots refers to test indices that are not scored: {missing}"
            )
        spot_rows = [lookup[value] for value in requested]

    gallery_rows: List[int] = []
    if not arguments.spots or arguments.gallery:
        gallery_rows = [int(row) for row in candidates["union"]]
        print("")
        print(f"[eval_spots] rendering the candidate gallery "
              f"({len(gallery_rows)} locations x 2 scales)...")
        for row in gallery_rows:
            for scale in ("linear", "db"):
                render_gallery_figure(output_dir, row, predictions, scale)
        print(f"[eval_spots] gallery written to "
              f"{os.path.join(output_dir, 'gallery', '{linear,db}')}")

    if spot_rows:
        print("")
        print(f"[eval_spots] final-figure mode: test indices "
              f"{[int(ground_truth.valid_indices[row]) for row in spot_rows]}")
        for scale in ("linear", "db"):
            render_spot_grid(output_dir, spot_rows, predictions, scale)
        print(f"[eval_spots] fig_qualitative_spots_{{linear,db}}.{{png,pdf}} written to "
              f"{output_dir}")

    # -- README -----------------------------------------------------------
    readme = [
        "eval_spots -- qualitative beam-pair maps where MIMO-GS and the MLP disagree",
        "=" * 70,
        "",
        "CONVENTIONS",
        "  Models      : fraction 1.0 only -- "
        f"{predictions['checkpoints'][METHOD_MIMOGS]},",
        f"                {predictions['checkpoints'][METHOD_MLP]},",
        "                plus the Sionna RT maps eval_baseline_rt.py uses.",
        "  Metric      : shape NMSE (max-normalized prediction vs. max-normalized",
        "                target) per location, in dB.  Imported from",
        "                evaluation/eval_render.py; never reimplemented.",
        f"  Test set    : the original full test.mat of {predictions['dataset_dir']};",
        f"                {ground_truth.num_scored} locations scored.",
        f"  Sionna RT   : {predictions['rt_note']}",
        "  Criteria    : (i)  NMSE gap  = NMSE(MLP) - NMSE(MIMO-GS), largest first.",
        "                (ii) top-4 gap = |top4(GT)\\top4(MLP)| - |top4(GT)\\top4(MIMO-GS)|,",
        "                     largest first; the miss counts come from",
        "                     eval_render.topk_metrics' K=4 overlap.",
        f"                Top {TOP_CANDIDATES_PER_CRITERION} under each criterion, deduped.",
        "  Scales      : 'linear' = each map divided by its own max, shared colorbar",
        f"                0..1.  'db' = 10*log10 of that, floored at {DB_FLOOR:.0f} dB.",
        f"  Device      : {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
        "  Figures     : no titles; axis labels 14 pt, ticks 12 pt; PNG at 300 dpi",
        "                plus PDF, gallery panels included.",
        "",
        "SELF-CHECK",
        f"  MIMO-GS model_100, full test set, this script : {measured:.4f} dB",
    ]
    if reference is None:
        readme.append("  eval_density reference                       : not found")
    else:
        readme += [
            f"  eval_density reference                       : {float(reference):.4f} dB",
            f"  delta                                        : "
            f"{abs(measured - float(reference)):.4f} dB "
            f"(tolerance {SELF_CHECK_TOLERANCE_DB:.2f} dB)",
        ]
    readme += [
        "",
        "HEADLINE NUMBERS (full test set, mean shape NMSE [dB])",
        f"  MIMO-GS      {float(np.mean(predictions['scored'][METHOD_MIMOGS]['nmse_shape_db'])):>10.3f}",
        f"  MLP          {float(np.mean(predictions['scored'][METHOD_MLP]['nmse_shape_db'])):>10.3f}",
    ]
    if rt_scored is not None:
        readme.append(
            f"  Sionna RT    {float(np.nanmean(rt_scored['nmse_shape_db'])):>10.3f}"
            f"   (matched locations only)"
        )
    readme += [
        "",
        f"CANDIDATES ({int(candidates['union'].size)} unique locations)",
        f"  largest NMSE gap    : {float(candidates['gap_nmse'][candidates['by_nmse']].max()):+.2f} dB",
        f"  largest top-4 gap   : {float(candidates['gap_top4'][candidates['by_top4']].max()):+.2f} beams",
        f"  overlap between the two criteria: "
        f"{len(by_nmse & by_top4)} location(s)",
        "",
        "FILES",
        "  candidates.csv                          both criteria + per-method NMSE",
        "  gallery/linear/loc_<index>.{png,pdf}    rows = GT / MIMO-GS / MLP / (RT)",
        "  gallery/db/loc_<index>.{png,pdf}        same, dB scale",
        "  fig_qualitative_spots_{linear,db}.*     final figure (--spots mode only)",
        "  README.txt                              this file",
        "",
        "WARNINGS",
    ]
    readme += [f"  {warning}" for warning in warnings] or ["  none"]
    readme += [
        "",
        "RERUN",
        "  python eval_spots.py",
        "  python eval_spots.py --spots <i>,<j>,<k>",
    ]
    write_readme(os.path.join(output_dir, "README.txt"), readme)

    print("")
    print("=" * 100)
    print("[eval_spots] SUMMARY")
    print("=" * 100)
    print(f"  {'method':<16}{'mean shape NMSE [dB]':>24}{'locations':>12}")
    print("  " + "-" * 52)
    print(f"  {METHOD_MIMOGS:<16}"
          f"{float(np.mean(predictions['scored'][METHOD_MIMOGS]['nmse_shape_db'])):>24.3f}"
          f"{ground_truth.num_scored:>12}")
    print(f"  {METHOD_MLP:<16}"
          f"{float(np.mean(predictions['scored'][METHOD_MLP]['nmse_shape_db'])):>24.3f}"
          f"{ground_truth.num_scored:>12}")
    if rt_scored is not None:
        print(f"  {METHOD_RT:<16}"
              f"{float(np.nanmean(rt_scored['nmse_shape_db'])):>24.3f}"
              f"{int(predictions['rt_matched_mask'].sum()):>12}")
    print("  " + "-" * 52)
    print(f"  candidates selected : {int(candidates['union'].size)} "
          f"(top {TOP_CANDIDATES_PER_CRITERION} per criterion, union)")
    print(f"  gallery figures     : {2 * len(gallery_rows)}")
    print(f"  final-figure spots  : "
          f"{[int(ground_truth.valid_indices[row]) for row in spot_rows] or 'not requested'}")
    print("")
    if warnings:
        print(f"[eval_spots] {len(warnings)} WARNING(S):")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("[eval_spots] No warnings.")
    print(f"[eval_spots] Outputs written to {output_dir}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
