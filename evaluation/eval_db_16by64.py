"""dB-scale ground-truth vs. rendered beam-map comparison for a 16x64 grid.

Runs with zero arguments::

    python eval_db_16by64.py

Linear-scale map figures only ever show the one or two dominant beam pairs;
everything 20 dB down is visually black.  This script renders the same maps on
a decibel scale so the weak multipath structure is actually inspectable, and
renders a random sample of test locations -- the same 50 that ``train.py``
saves to ``pred_compare/`` after training, drawn with the identical fixed seed
so the two sets of figures line up location for location.

The NMSE numbers are not recomputed here: ``eval_render.evaluate_test_set`` is
called directly, so the per-location values are by construction the same ones
E1 reports.  Only the handful of selected locations are re-rendered, for the
figures.

Power convention
----------------
The dataset field named ``magnitude`` is already a POWER map -- ``mean_b
|H_b|^2`` correlates 0.9986 with it at unit ratio (verified in
``eval_net_rate.py``).  The dB conversion is therefore ``10*log10(P / P.max())``
directly, NOT ``20*log10``: the maps are not amplitudes.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_baseline_rt import gain_net_hidden_dim, gain_net_width
from eval_render import (
    EPS,
    build_scene_and_model,
    evaluate_test_set,
    render_batch,
    resolve_run_dir,
    restore_config,
    summarize,
)
from utils.loss import normalize_mag_map


DEFAULT_CKPT = "outputs/20260805_051724"
DEFAULT_FLOOR_DB = -40.0
# train.py renders 50 random test samples into pred_compare/ with this seed;
# reusing both means the dB figures cover exactly the same locations.
DEFAULT_NUM_LOCATIONS = 50
TRAIN_SAMPLE_SEED = 12345
DEFAULT_OVERVIEW_ROWS = 8

# Reference values for this checkpoint, measured on the full test split.  The
# spot-check below compares against them and complains if the loading path has
# drifted; it does not silently accept a different model.
REFERENCE_NMSE_RAW_DB = -20.24
REFERENCE_NMSE_SHAPE_DB = -21.13
REFERENCE_TOLERANCE_DB = 0.1

# ----------------------------------------------------------------------
# Location selection
# ----------------------------------------------------------------------
def sample_locations(total_test: int, count: int, seed: int) -> List[int]:
    """The same random test indices ``train.py`` renders after training.

    ``train.evaluate_and_save_random_test_samples`` draws its sample with
    ``random.Random(12345).sample(range(len(test_set)), 50)``; mirroring that
    call means figure ``k`` here is the same UE as ``pred_compare/kk.png``.
    """
    count = min(int(count), int(total_test))
    return random.Random(int(seed)).sample(range(int(total_test)), count)


# ----------------------------------------------------------------------
# dB conversion and figures
# ----------------------------------------------------------------------
def to_db(power_map: np.ndarray, floor_db: float) -> np.ndarray:
    """Self-normalized power in dB, floored for display.

    Zeros are guarded before the log rather than after, so an all-zero row
    lands exactly on the floor instead of producing ``-inf``.
    """
    peak = float(power_map.max())
    if peak <= 0.0:
        return np.full_like(power_map, floor_db, dtype=np.float64)
    guarded = np.maximum(power_map / peak, 1e-30)
    return np.maximum(10.0 * np.log10(guarded), floor_db)


def plot_single_location(
    output_dir: str,
    stem: str,
    ground_truth: np.ndarray,
    rendered: np.ndarray,
    nmse_db: float,
    coordinates: np.ndarray,
    sample_label: str,
    floor_db: float,
) -> None:
    """Ground truth over rendering for one location, one shared colorbar."""
    # Constrained layout rather than tight_layout: the colorbar spans both
    # panels, which tight_layout cannot account for.
    figure, axes = plt.subplots(
        2, 1, figsize=(9.0, 5.2), sharex=True, layout="constrained"
    )

    image = None
    for axis, (title, power_map) in zip(
        axes, (("Ground truth", ground_truth), ("MIMO-GS rendering", rendered))
    ):
        image = axis.imshow(
            to_db(power_map, floor_db),
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=floor_db,
            vmax=0.0,
        )
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("Rx beam index", fontsize=9)
        axis.tick_params(labelsize=8)

    axes[-1].set_xlabel("Tx beam index", fontsize=9)

    colorbar = figure.colorbar(image, ax=axes.tolist(), fraction=0.035, pad=0.02)
    colorbar.set_label("Normalized power [dB]")

    figure.suptitle(
        f"{sample_label}  |  (x, y) = "
        f"({coordinates[0]:.1f}, {coordinates[1]:.1f}) m"
        f"  |  NMSE = {nmse_db:.2f} dB",
        fontsize=11,
    )

    figure.savefig(os.path.join(output_dir, f"{stem}.png"), dpi=200)
    figure.savefig(os.path.join(output_dir, f"{stem}.pdf"))
    plt.close(figure)


def plot_overview(
    output_dir: str,
    rows: int,
    ground_truth: np.ndarray,
    rendered: np.ndarray,
    nmse_db: np.ndarray,
    coordinates: np.ndarray,
    floor_db: float,
    total_rendered: int,
) -> None:
    """One row per location, GT | rendering, single shared colorbar.

    Only the first ``rows`` samples go on the overview: a 50-row version would
    be metres tall and unreadable.  Every sample still gets its own figure.
    """
    figure, axes = plt.subplots(
        rows,
        2,
        figsize=(11.0, 1.85 * rows + 1.2),
        squeeze=False,
        sharex=True,
        layout="constrained",
    )

    image = None
    for row in range(rows):
        for column, power_map in enumerate(
            (ground_truth[row], rendered[row])
        ):
            axis = axes[row][column]
            image = axis.imshow(
                to_db(power_map, floor_db),
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
                vmin=floor_db,
                vmax=0.0,
            )
            axis.tick_params(labelsize=7)
            if row == 0:
                axis.set_title(
                    ("Ground truth", "MIMO-GS rendering")[column], fontsize=10
                )
            if row == rows - 1:
                axis.set_xlabel("Tx beam index", fontsize=9)

        axes[row][0].set_ylabel(f"#{row:02d}\nRx beam", fontsize=8)
        axes[row][1].set_ylabel(
            f"NMSE {nmse_db[row]:.1f} dB\n"
            f"({coordinates[row, 0]:.0f}, {coordinates[row, 1]:.0f}) m",
            fontsize=8,
            rotation=270,
            labelpad=26,
            va="center",
        )
        axes[row][1].yaxis.set_label_position("right")

    colorbar = figure.colorbar(
        image, ax=axes.ravel().tolist(), fraction=0.02, pad=0.015
    )
    colorbar.set_label("Normalized power [dB]")

    figure.suptitle(
        f"Ground truth vs. MIMO-GS rendering on a dB scale  "
        f"(first {rows} of {total_rendered} random test locations, "
        f"floor {floor_db:g} dB, each map self-normalized)",
        fontsize=11,
    )

    figure.savefig(os.path.join(output_dir, "fig_overview.png"), dpi=200)
    figure.savefig(os.path.join(output_dir, "fig_overview.pdf"))
    plt.close(figure)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="dB-scale GT vs. rendered beam-map figures (16x64)"
    )
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--floor_db", type=float, default=DEFAULT_FLOOR_DB)
    parser.add_argument(
        "--num_locations",
        type=int,
        default=DEFAULT_NUM_LOCATIONS,
        help="How many random test locations to render (train.py uses 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=TRAIN_SAMPLE_SEED,
        help="Sampling seed; the default matches train.py's pred_compare set.",
    )
    parser.add_argument(
        "--overview_rows",
        type=int,
        default=DEFAULT_OVERVIEW_ROWS,
        help="Locations shown on the combined overview figure.",
    )
    parser.add_argument("--outputs_root", type=str, default="outputs")
    parser.add_argument("--analysis_root", type=str, default="analysis")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--source_path", type=str, default="")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    repository_root = os.path.dirname(os.path.abspath(__file__))

    floor_db = float(arguments.floor_db)
    if floor_db >= 0.0:
        raise SystemExit(
            f"[eval_db_16by64] --floor_db must be negative, got {floor_db:g}"
        )

    outputs_root = arguments.outputs_root
    if not os.path.isabs(outputs_root):
        outputs_root = os.path.join(repository_root, outputs_root)

    run_dir, checkpoint_path = resolve_run_dir(arguments.ckpt, outputs_root)
    run_name = os.path.basename(os.path.normpath(run_dir))

    print("=" * 78)
    print(f"[eval_db_16by64] RUN        : {run_name}")
    print(f"[eval_db_16by64] checkpoint : {checkpoint_path}")
    print(f"[eval_db_16by64] floor      : {floor_db:g} dB | "
          f"locations: {arguments.num_locations}")
    print("=" * 78)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_params, opt_params = restore_config(run_dir, checkpoint)

    if arguments.source_path:
        model_params.source_path = os.path.abspath(arguments.source_path)
    if not os.path.isdir(getattr(model_params, "source_path", "")):
        raise SystemExit(
            f"[eval_db_16by64] Dataset "
            f"'{getattr(model_params, 'source_path', '')}' is missing."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda_rasterizer = (
        bool(int(getattr(model_params, "use_cuda_rasterizer", 1)))
        and device.type == "cuda"
    )
    batch_size = max(
        1, int(arguments.batch_size) or int(getattr(model_params, "batch_size", 8))
    )

    # This checkpoint's gain MLP is narrower than the current repo default, so
    # it is rebuilt at the stored width before restore -- the same adaptation
    # eval_baseline_rt.py makes.
    hidden_dim = gain_net_hidden_dim(checkpoint)
    if hidden_dim is not None:
        print(
            f"[eval_db_16by64] checkpoint gain MLP is {hidden_dim}-wide "
            f"(repo default differs); rebuilding it to match."
        )
    with gain_net_width(hidden_dim):
        scene, gaussians = build_scene_and_model(
            model_params, opt_params, checkpoint, device
        )

    scale_factor = float(getattr(scene.test_set, "scale_factor", 1.0))
    print(
        f"[eval_db_16by64] device={device} | batch_size={batch_size} | "
        f"beam grid {scene.beam_rows}x{scene.beam_cols} | "
        f"test samples={len(scene.test_set)}"
    )

    # ------------------------------------------------------------------
    # NMSE straight from eval_render, so the numbers cannot drift from E1
    # ------------------------------------------------------------------
    print("[eval_db_16by64] scoring the test split with eval_render ...")
    results = evaluate_test_set(
        scene, gaussians, model_params, device, batch_size, use_cuda_rasterizer
    )

    nmse_raw = results["nmse_raw_db"]
    nmse_shape = results["nmse_shape_db"]
    coordinates = results["position"] * scale_factor
    dataset_indices = results["index"]
    skipped = int(results["skipped_zero_power"])

    raw_stats = summarize(nmse_raw)
    shape_stats = summarize(nmse_shape)
    print(
        f"[eval_db_16by64] evaluated {nmse_raw.size} locations "
        f"(skipped zero-power: {skipped})"
    )
    print(
        f"  NMSE_raw   mean {raw_stats['mean']:.2f} dB / median "
        f"{raw_stats['median']:.2f} dB"
    )
    print(
        f"  NMSE_shape mean {shape_stats['mean']:.2f} dB / median "
        f"{shape_stats['median']:.2f} dB"
    )

    for name, measured, reference in (
        ("NMSE_raw", raw_stats["mean"], REFERENCE_NMSE_RAW_DB),
        ("NMSE_shape", shape_stats["mean"], REFERENCE_NMSE_SHAPE_DB),
    ):
        delta = abs(measured - reference)
        verdict = "OK" if delta <= REFERENCE_TOLERANCE_DB else "MISMATCH"
        print(
            f"  spot-check {name}: {measured:.2f} dB vs expected "
            f"{reference:.2f} dB (delta {delta:.3f}) : {verdict}"
        )
        if delta > REFERENCE_TOLERANCE_DB:
            raise SystemExit(
                f"[eval_db_16by64] {name} mean is {measured:.2f} dB but this "
                f"checkpoint should give {reference:.2f} dB. The loading path "
                f"differs from eval_render's; fix this script."
            )

    # ------------------------------------------------------------------
    # Random location sample (train.py's pred_compare set) + re-render
    # ------------------------------------------------------------------
    figure_indices = sample_locations(
        len(scene.test_set), int(arguments.num_locations), int(arguments.seed)
    )
    print(
        f"[eval_db_16by64] sampling {len(figure_indices)} random test locations "
        f"with seed {arguments.seed}"
        + (
            "  (identical to train.py's pred_compare set)"
            if int(arguments.seed) == TRAIN_SAMPLE_SEED
            else ""
        )
    )

    # evaluate_test_set drops zero-power maps, so its row order is not the
    # dataset order; map dataset index -> results row to attach the NMSE.
    row_of_index = {int(v): row for row, v in enumerate(dataset_indices)}
    missing = [i for i in figure_indices if i not in row_of_index]
    if missing:
        print(
            f"[eval_db_16by64] {len(missing)} sampled location(s) were skipped as "
            f"zero-power and are dropped from the figures."
        )
        figure_indices = [i for i in figure_indices if i in row_of_index]

    result_rows = [row_of_index[i] for i in figure_indices]

    ground_truth = torch.stack(
        [
            scene.test_set[i][0].reshape(scene.beam_rows, scene.beam_cols)
            for i in figure_indices
        ],
        dim=0,
    ).to(device)
    rx_positions = torch.stack(
        [scene.test_set[i][1].reshape(3) for i in figure_indices], dim=0
    ).to(device)
    tx_pos = torch.as_tensor(scene.bs_position, dtype=torch.float32, device=device)

    rendered_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, rx_positions.shape[0], batch_size):
            rendered_chunks.append(
                render_batch(
                    rx_positions[start : start + batch_size],
                    tx_pos,
                    gaussians,
                    scene,
                    model_params,
                    use_cuda_rasterizer,
                ).float()
            )
    rendered = torch.cat(rendered_chunks, dim=0)

    # Both panels are self-normalized for display, so the comparison is of
    # shape rather than of absolute level.
    ground_truth_np = ground_truth.detach().cpu().numpy().astype(np.float64)
    rendered_np = rendered.detach().cpu().numpy().astype(np.float64)

    selected_nmse = np.asarray(
        [nmse_raw[row] for row in result_rows], dtype=np.float64
    )
    selected_coordinates = np.stack(
        [coordinates[row] for row in result_rows], axis=0
    )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    output_dir = os.path.join(run_dir, "db_maps")
    os.makedirs(output_dir, exist_ok=True)

    records: List[Dict[str, object]] = []
    for sample, (dataset_index, row) in enumerate(zip(figure_indices, result_rows)):
        gt_map = ground_truth_np[sample]
        positive = gt_map[gt_map > 0.0]
        dynamic_range = float(
            10.0
            * np.log10(
                max(float(gt_map.max()), EPS)
                / max(float(positive.min()) if positive.size else EPS, 1e-30)
            )
        )
        stem = f"fig_loc_{sample:02d}"
        plot_single_location(
            output_dir,
            stem,
            gt_map,
            rendered_np[sample],
            float(nmse_raw[row]),
            selected_coordinates[sample],
            f"sample #{sample:02d} (test index {dataset_index})",
            floor_db,
        )
        records.append(
            {
                "sample": sample,
                "figure": f"{stem}.png",
                "test_index": int(dataset_index),
                "x": f"{selected_coordinates[sample, 0]:.6f}",
                "y": f"{selected_coordinates[sample, 1]:.6f}",
                "z": f"{selected_coordinates[sample, 2]:.6f}",
                "NMSE_raw_dB": f"{nmse_raw[row]:.6f}",
                "NMSE_shape_dB": f"{nmse_shape[row]:.6f}",
                "gt_dynamic_range_dB": f"{dynamic_range:.2f}",
            }
        )

    overview_rows = max(1, min(int(arguments.overview_rows), len(figure_indices)))
    plot_overview(
        output_dir,
        overview_rows,
        ground_truth_np,
        rendered_np,
        selected_nmse,
        selected_coordinates,
        floor_db,
        len(figure_indices),
    )

    with open(
        os.path.join(output_dir, "selected_locations.csv"), "w", newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    sample_stats = summarize(selected_nmse)
    print()
    print(
        f"[eval_db_16by64] rendered {len(figure_indices)} locations | "
        f"NMSE over this sample: mean {sample_stats['mean']:.2f} dB, median "
        f"{sample_stats['median']:.2f} dB, best {selected_nmse.min():.2f}, "
        f"worst {selected_nmse.max():.2f}"
    )
    print(f"[eval_db_16by64] overview figure shows the first {overview_rows}")

    # ------------------------------------------------------------------
    # How much of each map the floor actually hides
    # ------------------------------------------------------------------
    print()
    print(
        f"[eval_db_16by64] fraction of GT bins below the floor "
        f"(what a {floor_db:g} dB floor clips away)"
    )
    for candidate_floor in (-20.0, -30.0, -40.0, -50.0):
        below_gt = float(
            np.mean(
                [
                    np.mean(
                        10.0
                        * np.log10(np.maximum(m / max(m.max(), 1e-30), 1e-30))
                        < candidate_floor
                    )
                    for m in ground_truth_np
                ]
            )
        )
        below_pred = float(
            np.mean(
                [
                    np.mean(
                        10.0
                        * np.log10(np.maximum(m / max(m.max(), 1e-30), 1e-30))
                        < candidate_floor
                    )
                    for m in rendered_np
                ]
            )
        )
        marker = "  <-- active" if abs(candidate_floor - floor_db) < 1e-9 else ""
        print(
            f"  floor {candidate_floor:>6.0f} dB: GT {100.0 * below_gt:5.1f}% of "
            f"bins clipped, rendering {100.0 * below_pred:5.1f}%{marker}"
        )

    print()
    print(f"[eval_db_16by64] Outputs written to {output_dir}")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
