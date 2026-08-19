#!/usr/bin/env python3
"""D2 -- rendering fidelity vs. distance from the training locations.

Uses the fraction-1.0 density checkpoints only::

    outputs/density/mimogs/model_100.pth
    outputs/density/MLP/model_100.pth

plus the learning-free nearest-neighbour baseline on the FULL train set, and
bins the per-test-location shape NMSE by how far that location sits from the
training data.  Outputs land in ``analysis/eval_distance/``.

Zero-argument runnable::

    python eval_distance.py

Nothing in the repository is modified.  All metric code, checkpoint loading and
figure conventions are imported from ``eval_density.py``, which in turn imports
every metric from ``evaluation/eval_render.py``.

A note on the x axis
--------------------
The requested binning variable is the distance from each test location to its
NEAREST training location.  On an interleaved train/test lattice -- which is
what this dataset ships -- that quantity is very nearly constant, so it cannot
be split into 5-8 bins of >= 100 locations.  The script always computes and
reports it, prints a WARN when it degenerates, and then escalates to the
distance to the k-th nearest training location (the local training-support
radius, same units, still "distance to the training data") for the smallest k
on a fixed ladder that does admit a valid binning.  Both quantities are in the
CSV and the chosen k is stated on the figure axis and in this README.
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
    LEGEND_FONTSIZE,
    LEGEND_ORDER,
    METHOD_MIMOGS,
    METHOD_MLP,
    METHOD_NN,
    METHOD_STYLE,
    REPO_ROOT,
    TICK_LABELSIZE,
    TestGroundTruth,
    assert_finite_nonnegative,
    load_mimogs,
    load_mlp,
    load_train_mat,
    nearest_neighbour_indices,
    nearest_neighbour_maps,
    predict_mlp_maps,
    render_mimogs_maps,
    resolve_device,
    save_figure,
    style_axis,
    write_csv,
    write_readme,
)


MIN_BIN_COUNT = 100
DESIRED_BIN_COUNTS: Tuple[int, ...] = (8, 7, 6, 5)
MIN_BINS = 5
MAX_BINS = 8

# Smallest-first ladder of "distance to the k-th nearest training location".
# k = 1 is the requested definition; the rest are the documented escalation.
NEIGHBOUR_LADDER: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

METHODS: Tuple[str, ...] = (METHOD_MIMOGS, METHOD_MLP, METHOD_NN)


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------
def greedy_value_bins(
    values: np.ndarray, desired_bins: int, min_count: int = MIN_BIN_COUNT
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin a heavily discretized quantity by merging adjacent distinct values.

    Quantile edges are useless here: positions live on a lattice, so the
    distances take only a handful of distinct values and several quantiles
    collapse onto the same number.  Adjacent distinct values are therefore
    accumulated until a bin reaches its target population, and the first/last
    bins are merged inward if they still fall short.

    Returns ``(edges, counts)`` with ``len(edges) == len(counts) + 1``.
    """
    unique, occurrences = np.unique(np.round(np.asarray(values, dtype=np.float64), 9),
                                    return_counts=True)
    total = int(occurrences.sum())
    target = max(int(min_count), int(np.ceil(total / max(1, int(desired_bins)))))

    groups: List[Tuple[List[int], int]] = []
    current: List[int] = []
    running = 0

    for position in range(unique.size):
        current.append(position)
        running += int(occurrences[position])
        last_value = position == unique.size - 1
        if running >= target and len(groups) < int(desired_bins) - 1 and not last_value:
            groups.append((current, running))
            current, running = [], 0
    if current:
        groups.append((current, running))

    # Merge an underfull edge bin into its neighbour rather than reporting it.
    while len(groups) > 1 and groups[-1][1] < min_count:
        head, tail = groups[-2], groups[-1]
        groups[-2] = (head[0] + tail[0], head[1] + tail[1])
        groups.pop()
    while len(groups) > 1 and groups[0][1] < min_count:
        head, tail = groups[0], groups[1]
        groups[0] = (head[0] + tail[0], head[1] + tail[1])
        groups.pop(1)

    edges = [float(unique[groups[0][0][0]])]
    for index in range(len(groups) - 1):
        upper = float(unique[groups[index][0][-1]])
        lower = float(unique[groups[index + 1][0][0]])
        edges.append(0.5 * (upper + lower))
    edges.append(float(unique[groups[-1][0][-1]]))

    return np.asarray(edges, dtype=np.float64), np.asarray(
        [group[1] for group in groups], dtype=np.int64
    )


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index per value, clamped to the outermost bins."""
    return np.clip(
        np.digitize(np.asarray(values, dtype=np.float64), edges[1:-1], right=False),
        0,
        edges.size - 2,
    ).astype(np.int64)


def try_binning(values: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """First feasible ``(edges, counts)`` over the desired bin-count ladder."""
    for desired in DESIRED_BIN_COUNTS:
        edges, counts = greedy_value_bins(values, desired)
        if MIN_BINS <= counts.size <= MAX_BINS and int(counts.min()) >= MIN_BIN_COUNT:
            return edges, counts
    return None


def select_binning_variable(
    test_positions: np.ndarray, train_positions: np.ndarray
) -> Dict[str, object]:
    """Pick the smallest k whose k-th-nearest-train distance bins cleanly."""
    largest_k = min(max(NEIGHBOUR_LADDER), int(train_positions.shape[0]))
    all_distances, _ = nearest_neighbour_indices(
        test_positions, train_positions, k=largest_k
    )

    attempts: List[Dict[str, object]] = []
    for k in NEIGHBOUR_LADDER:
        if k > largest_k:
            break
        values = all_distances[:, k - 1].astype(np.float64)
        binning = try_binning(values)
        attempts.append(
            {
                "k": int(k),
                "num_distinct": int(np.unique(np.round(values, 9)).size),
                "feasible": binning is not None,
                "counts": None if binning is None else binning[1].tolist(),
            }
        )
        if binning is not None:
            edges, counts = binning
            return {
                "k": int(k),
                "values": values,
                "edges": edges,
                "counts": counts,
                "attempts": attempts,
                "nearest_distance": all_distances[:, 0].astype(np.float64),
            }

    # Nothing on the ladder worked: fall back to the requested definition and
    # let the caller report the degeneracy honestly.
    values = all_distances[:, 0].astype(np.float64)
    edges, counts = greedy_value_bins(values, DESIRED_BIN_COUNTS[-1])
    return {
        "k": 1,
        "values": values,
        "edges": edges,
        "counts": counts,
        "attempts": attempts,
        "nearest_distance": values,
        "failed": True,
    }


def axis_label_for(k: int) -> str:
    if int(k) == 1:
        return "Distance to nearest training location [m]"
    return f"Distance to the {int(k)}-th nearest training location [m]"


# ---------------------------------------------------------------------------
# Per-location scoring
# ---------------------------------------------------------------------------
def score_all_methods(
    arguments: argparse.Namespace, device: torch.device
) -> Dict[str, object]:
    """Per-test-location shape NMSE for MIMO-GS / MLP / nearest neighbour."""
    mimogs_path = os.path.join(arguments.mimogs_dir, "model_100.pth")
    mlp_path = os.path.join(arguments.mlp_dir, "model_100.pth")

    probe = torch.load(mimogs_path, map_location="cpu", weights_only=False)
    dataset_dir = os.path.abspath(arguments.dataset or probe["config"]["dataset_path"])
    del probe

    ground_truth = TestGroundTruth(dataset_dir, device)
    train_positions, train_magnitude = load_train_mat(dataset_dir)

    loaded_gs = load_mimogs(mimogs_path, device, dataset_dir)
    gs_maps = render_mimogs_maps(loaded_gs, ground_truth.positions_normalized)
    assert_finite_nonnegative(gs_maps, "MIMO-GS model_100")
    gs_scored = ground_truth.score(gs_maps)

    loaded_mlp = load_mlp(mlp_path, device)
    mlp_maps = predict_mlp_maps(loaded_mlp, ground_truth.positions_normalized)
    assert_finite_nonnegative(mlp_maps, "MLP model_100")
    mlp_scored = ground_truth.score(mlp_maps)

    nn_maps, nn_distance = nearest_neighbour_maps(
        train_positions, train_magnitude, ground_truth.positions_m, device
    )
    assert_finite_nonnegative(nn_maps, "Nearest neighbor (full train set)")
    nn_scored = ground_truth.score(nn_maps)

    return {
        "dataset_dir": dataset_dir,
        "ground_truth": ground_truth,
        "train_positions": train_positions,
        "nmse": {
            METHOD_MIMOGS: gs_scored["nmse_shape_db"],
            METHOD_MLP: mlp_scored["nmse_shape_db"],
            METHOD_NN: nn_scored["nmse_shape_db"],
        },
        "nn_distance_full": nn_distance,
        "checkpoints": {
            METHOD_MIMOGS: os.path.relpath(mimogs_path, REPO_ROOT),
            METHOD_MLP: os.path.relpath(mlp_path, REPO_ROOT),
            METHOD_NN: "(no learning, full train set)",
        },
        "n_train": int(train_positions.shape[0]),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_nmse_vs_distance(
    output_dir: str,
    edges: np.ndarray,
    centers: np.ndarray,
    per_bin: Dict[str, np.ndarray],
    counts: np.ndarray,
    xlabel: str,
) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.6))

    count_axis = axis.twinx()
    count_axis.bar(
        centers,
        counts,
        width=0.85 * np.diff(edges),
        color="0.85",
        edgecolor="0.65",
        linewidth=0.5,
        zorder=0,
    )
    count_axis.set_ylabel("Test locations per bin", fontsize=AXIS_LABEL_FONTSIZE)
    count_axis.tick_params(labelsize=TICK_LABELSIZE)
    count_axis.set_ylim(0, float(counts.max()) * 3.2)
    count_axis.set_zorder(0)

    axis.set_zorder(1)
    axis.patch.set_visible(False)

    for method in METHODS:
        style = METHOD_STYLE[method]
        axis.plot(
            centers,
            per_bin[method],
            label=method,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=5.5,
            zorder=3,
        )

    style_axis(axis, xlabel, "Shape NMSE [dB]")
    axis.grid(alpha=0.3, linewidth=0.5)

    handles, labels = axis.get_legend_handles_labels()
    ordered = [(handles[labels.index(name)], name) for name in LEGEND_ORDER if name in labels]
    axis.legend(
        [handle for handle, _ in ordered],
        [name for _, name in ordered],
        fontsize=LEGEND_FONTSIZE,
        loc="best",
    )

    save_figure(figure, output_dir, "fig_nmse_vs_distance")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="D2 -- shape NMSE vs. distance from the training locations"
    )
    parser.add_argument("--mimogs_dir", type=str, default=DEFAULT_MIMOGS_DIR)
    parser.add_argument("--mlp_dir", type=str, default=DEFAULT_MLP_DIR)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--analysis_root", type=str, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    return parser


def main() -> int:
    arguments = build_argument_parser().parse_args()
    device = resolve_device(prefer_cuda=not arguments.cpu)
    warnings: List[str] = []

    print("=" * 100)
    print("[eval_distance] Shape NMSE vs. distance from the training locations")
    print("=" * 100)
    print(f"[eval_distance] device : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    scored = score_all_methods(arguments, device)
    ground_truth: TestGroundTruth = scored["ground_truth"]
    positions = ground_truth.valid_positions_m
    train_positions = scored["train_positions"]

    print(f"[eval_distance] dataset: {scored['dataset_dir']}")
    print(f"[eval_distance] scored test locations: {ground_truth.num_scored} "
          f"(skipped zero-power: {ground_truth.num_skipped_zero_power})")
    print(f"[eval_distance] train locations      : {scored['n_train']}")
    print("")
    for method in METHODS:
        print(f"[eval_distance] {method:<18} full-test mean shape NMSE = "
              f"{float(np.mean(scored['nmse'][method])):8.3f} dB")

    # -- binning variable ------------------------------------------------
    selection = select_binning_variable(positions, train_positions)
    nearest_distance = selection["nearest_distance"]
    binning_values = selection["values"]
    edges = selection["edges"]
    counts = selection["counts"]
    chosen_k = int(selection["k"])

    print("")
    print("-" * 100)
    print("[eval_distance] BINNING")
    print("-" * 100)
    print(f"  requested variable: distance to the nearest training location")
    print(f"    min / median / max = {nearest_distance.min():.4f} / "
          f"{np.median(nearest_distance):.4f} / {nearest_distance.max():.4f} m, "
          f"{int(np.unique(np.round(nearest_distance, 9)).size)} distinct values")

    if not bool(selection.get("failed", False)) and chosen_k != 1:
        warnings.append(
            "WARN the requested binning variable (distance to the NEAREST training "
            "location) is degenerate on this split: the train/test grids are "
            f"interleaved, so {100.0 * float(np.mean(np.isclose(nearest_distance, np.median(nearest_distance)))):.1f}% "
            f"of the test locations sit at exactly {np.median(nearest_distance):.3f} m "
            "from a training location and no 5-8 bin split with >= 100 locations "
            f"each exists.  Escalated to the {chosen_k}-th nearest training "
            "location (local training-support radius, same units)."
        )
    if bool(selection.get("failed", False)):
        warnings.append(
            "WARN no k on the neighbour ladder produced 5-8 bins with >= 100 "
            "locations each; the figure falls back to the nearest-neighbour "
            "distance with whatever bins the data admits."
        )

    for attempt in selection["attempts"]:
        print(f"    k={int(attempt['k']):>4}  distinct={int(attempt['num_distinct']):>4}  "
              f"feasible={'yes' if attempt['feasible'] else 'no '}"
              + (f"  counts={attempt['counts']}" if attempt["feasible"] else ""))

    print(f"  chosen variable   : {axis_label_for(chosen_k)}")
    print(f"  bin edges [m]     : {[round(float(e), 4) for e in edges]}")
    print(f"  bin counts        : {counts.tolist()}")

    bin_index = assign_bins(binning_values, edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    per_bin: Dict[str, np.ndarray] = {}
    for method in METHODS:
        values = scored["nmse"][method]
        per_bin[method] = np.asarray(
            [float(np.mean(values[bin_index == b])) for b in range(counts.size)],
            dtype=np.float64,
        )

    print("")
    header = f"  {'bin':>4}{'range [m]':>22}{'count':>8}" + "".join(
        f"{method:>18}" for method in METHODS
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for b in range(counts.size):
        print(
            f"  {b:>4}{f'[{edges[b]:.3f}, {edges[b + 1]:.3f}]':>22}{int(counts[b]):>8}"
            + "".join(f"{per_bin[method][b]:>18.3f}" for method in METHODS)
        )
    print("  " + "-" * (len(header) - 2))

    # -- outputs ---------------------------------------------------------
    output_dir = os.path.join(arguments.analysis_root, "eval_distance")
    os.makedirs(output_dir, exist_ok=True)

    plot_nmse_vs_distance(
        output_dir, edges, centers, per_bin, counts, axis_label_for(chosen_k)
    )

    per_location_header = [
        "test_index",
        "x_m",
        "y_m",
        "z_m",
        "nearest_train_distance_m",
        f"k{chosen_k}_train_distance_m",
        "bin_index",
        "bin_low_m",
        "bin_high_m",
        "nmse_shape_dB_mimogs",
        "nmse_shape_dB_mlp",
        "nmse_shape_dB_nearest_neighbor",
    ]
    per_location_rows = []
    for row in range(ground_truth.num_scored):
        b = int(bin_index[row])
        per_location_rows.append(
            [
                int(ground_truth.valid_indices[row]),
                f"{positions[row, 0]:.6f}",
                f"{positions[row, 1]:.6f}",
                f"{positions[row, 2]:.6f}",
                f"{nearest_distance[row]:.6f}",
                f"{binning_values[row]:.6f}",
                b,
                f"{edges[b]:.6f}",
                f"{edges[b + 1]:.6f}",
                f"{scored['nmse'][METHOD_MIMOGS][row]:.6f}",
                f"{scored['nmse'][METHOD_MLP][row]:.6f}",
                f"{scored['nmse'][METHOD_NN][row]:.6f}",
            ]
        )
    write_csv(
        os.path.join(output_dir, "per_location.csv"), per_location_header, per_location_rows
    )

    bin_header = ["bin_index", "bin_low_m", "bin_high_m", "bin_center_m", "count"] + [
        f"mean_nmse_shape_dB_{method.lower().replace(' ', '_').replace('-', '')}"
        for method in METHODS
    ]
    bin_rows = [
        [
            b,
            f"{edges[b]:.6f}",
            f"{edges[b + 1]:.6f}",
            f"{centers[b]:.6f}",
            int(counts[b]),
        ]
        + [f"{per_bin[method][b]:.6f}" for method in METHODS]
        for b in range(counts.size)
    ]
    write_csv(os.path.join(output_dir, "distance_bins.csv"), bin_header, bin_rows)

    readme = [
        "eval_distance -- shape NMSE vs. distance from the training locations",
        "=" * 70,
        "",
        "CONVENTIONS",
        "  Models      : fraction 1.0 only -- outputs/density/mimogs/model_100.pth,",
        "                outputs/density/MLP/model_100.pth, and the nearest-neighbour",
        "                baseline over the FULL train set.",
        "  Metric      : shape NMSE (max-normalized prediction vs. max-normalized",
        "                target) per location, in dB.  Imported from",
        "                evaluation/eval_render.py; never reimplemented.",
        f"  Test set    : the original full test.mat of {scored['dataset_dir']};",
        f"                {ground_truth.num_scored} locations scored, "
        f"{ground_truth.num_skipped_zero_power} skipped for zero power.",
        f"  Train set   : {scored['n_train']} locations.",
        "  Distances   : 3D Euclidean, in ORIGINAL meters (never the per-file",
        "                normalized coordinates).",
        f"  Device      : {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
        "  Figures     : no titles; axis labels 14 pt, ticks 12 pt, legend 10 pt;",
        "                PNG at 300 dpi plus PDF.  The grey bars on the right-hand",
        "                axis are the per-bin location counts.",
        "",
        "BINNING VARIABLE",
        "  Requested: distance to the NEAREST training location.",
        f"    min / median / max = {nearest_distance.min():.4f} / "
        f"{np.median(nearest_distance):.4f} / {nearest_distance.max():.4f} m",
        f"    distinct values    = {int(np.unique(np.round(nearest_distance, 9)).size)}",
        "  This dataset's train/test split is an interleaved lattice, so that",
        "  quantity is essentially constant and admits no 5-8 bin split with >= 100",
        "  locations per bin.  The script therefore escalates along a ladder of",
        "  k-th-nearest-training-location distances (same units, same 'distance to",
        "  the training data' meaning, but sensitive to local sampling density) and",
        "  uses the smallest feasible k.",
        f"  Chosen: k = {chosen_k}  ->  {axis_label_for(chosen_k)}",
        "  Ladder attempts (k, distinct values, feasible):",
    ]
    for attempt in selection["attempts"]:
        readme.append(
            f"    k={int(attempt['k']):>4}  distinct={int(attempt['num_distinct']):>4}  "
            f"feasible={'yes' if attempt['feasible'] else 'no'}"
        )
    readme += [
        "",
        "BIN EDGES AND COUNTS",
        f"  {'bin':>4}{'range [m]':>22}{'count':>8}"
        + "".join(f"{method:>18}" for method in METHODS),
    ]
    for b in range(counts.size):
        readme.append(
            f"  {b:>4}{f'[{edges[b]:.3f}, {edges[b + 1]:.3f}]':>22}{int(counts[b]):>8}"
            + "".join(f"{per_bin[method][b]:>18.3f}" for method in METHODS)
        )
    readme += [
        "",
        "HEADLINE NUMBERS (full test set, mean shape NMSE [dB])",
    ]
    for method in METHODS:
        readme.append(
            f"  {method:<20}{float(np.mean(scored['nmse'][method])):>10.3f}"
            f"   ({scored['checkpoints'][method]})"
        )
    readme += [
        "",
        "FILES",
        "  fig_nmse_vs_distance.{png,pdf}  mean shape NMSE per distance bin",
        "  per_location.csv                x, y, z, nearest-train distance, binning",
        "                                  distance, bin, NMSE per method",
        "  distance_bins.csv               per-bin edges, counts and means",
        "  README.txt                      this file",
        "",
        "WARNINGS",
    ]
    readme += [f"  {warning}" for warning in warnings] or ["  none"]
    readme += ["", "RERUN", "  python eval_distance.py"]

    write_readme(os.path.join(output_dir, "README.txt"), readme)

    print("")
    print("=" * 100)
    print("[eval_distance] SUMMARY")
    print("=" * 100)
    print(f"  {'method':<20}{'full-test mean [dB]':>22}{'best bin [dB]':>16}{'worst bin [dB]':>16}")
    print("  " + "-" * 72)
    for method in METHODS:
        print(
            f"  {method:<20}{float(np.mean(scored['nmse'][method])):>22.3f}"
            f"{float(per_bin[method].min()):>16.3f}{float(per_bin[method].max()):>16.3f}"
        )
    print("  " + "-" * 72)
    print(f"  binning variable : {axis_label_for(chosen_k)}")
    print(f"  bins             : {counts.size} "
          f"(counts {counts.tolist()}, min {int(counts.min())})")
    print("")
    if warnings:
        print(f"[eval_distance] {len(warnings)} WARNING(S):")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("[eval_distance] No warnings.")
    print(f"[eval_distance] Outputs written to {output_dir}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
