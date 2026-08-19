#!/usr/bin/env python3
"""Pure coordinate-MLP baseline on the indoor 63x63 dataset (1000 epochs).

Zero-argument runnable::

    python train_MLP_indoor.py

This is the ``measured_custom_beams`` branch counterpart of
``evaluation/train_MLP.py``: the model, the loss, the metrics and the
train/test split all come from that module, so the numbers printed here are
directly comparable with a ``train.py`` MIMO-GS run on the same dataset.
Only the run configuration is pinned differently:

* model    -- the ``mlp_medium`` configuration only (hidden 512, depth 6).
* dataset  -- ``dataset/indoor_63by63`` (275 train / 75 test, 63 x 63 map).
  The 63-beam grid is the measured 21 az x 3 el steering codebook, but the MLP
  never looks at beam geometry: it regresses the flattened 3969-bin map
  directly, so ``beam_grid_mode`` only matters for the Gaussian renderer.
* epochs   -- 1000 (the ``ModelParams`` default on this branch), against the
  10 epochs ``evaluation/train_MLP.py`` inherits for the DeepMIMO datasets.
  275 samples at batch size 8 is 35 iterations per epoch, so a full run is
  35k iterations.
* logging  -- the test set is evaluated every ``--eval_every`` epochs instead
  of every epoch, and the best epoch (lowest per-location shape NMSE) is kept
  alongside the final one. With 1000 epochs on 275 samples the final epoch is
  not necessarily the best one.

``evaluation/train_MLP.py`` cannot be launched as a file (it imports
``arguments`` / ``scene`` as top-level modules without putting the repo root on
``sys.path``); running this script from the repo root resolves those imports.

Outputs land in ``outputs/mlp_indoor_medium/`` (kept separate from the
``outputs/mlp_indoor/`` directory an earlier three-config run wrote):

* ``model.pth``     -- final + best state dicts, config, full trajectory
* ``config.json``   -- the same metadata without the tensors
* ``pred_compare/``    -- ground-truth vs predicted figures, linear scale
* ``pred_compare_db/`` -- the same 50 samples in dB

The figures follow ``train.evaluate_and_save_random_test_samples``: same fixed
sampling seed (12345), so the selected test locations are the ones a ``train.py``
run renders, and the same two-panel layout (max-normalized ground truth on top,
raw prediction below).  Both directories are written from one forward pass and
share file names, so ``pred_compare/07.png`` and ``pred_compare_db/07.png`` show
the same test location.  The dB view is ``10 * log10`` over a fixed -50..0 dB
window, the convention of the commented-out power-domain branch in train.py.

Metrics, all on the held-out test set:

* ``per-loc shape``  -- per-location dB of the normalized-prediction NMSE,
  averaged in dB. The headline column ``eval_render.py`` reports.
* ``per-loc raw``    -- same, for the raw (unnormalized) prediction.
* ``scale``/``shape``-- linear NMSE averaged over the test set and converted to
  dB once, the convention ``train.evaluate_full_test_quality`` prints.
* ``top-1``          -- fraction of test locations whose predicted argmax beam
  pair equals the ground-truth argmax beam pair.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from argparse import ArgumentParser
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # figures are written to disk, never shown

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from arguments import ModelParams
from evaluation.train_MLP import (
    BASE_LR,
    FINAL_LR,
    LOSS_EPS,
    SEED,
    TOPK_RATIO,
    PositionMLP,
    build_scene,
    count_parameters,
    evaluate_test_nmse,
)
from utils.loss import composite_magnitude_loss, normalize_mag_map


DATASET_DIR = "./dataset/indoor_63by63"
EPOCHS = 1000
BATCH_SIZE = 8
EVAL_EVERY = 10
OUTPUT_DIR = os.path.join("outputs", "mlp_indoor_medium")

# The ``mlp_medium`` configuration of evaluation/train_MLP.py.
HIDDEN = 512
DEPTH = 6

# Post-training figure settings, mirroring train.py.
NUM_EVAL_SAMPLES = 50
EVAL_BATCH_SIZE = 50
SAMPLE_SEED = 12345

# dB-scale figures, following the commented-out power-domain branch of
# train.evaluate_and_save_random_test_samples.  The fixed window makes the two
# panels of a figure, and every figure of a run, directly comparable.
DB_EPS = 1e-12
DB_VMIN = -50.0
DB_VMAX = 0.0


@torch.no_grad()
def evaluate_top1_accuracy(model: PositionMLP, scene, device: torch.device) -> float:
    """Fraction of test locations whose predicted best beam pair is correct."""
    model.eval()

    hits = 0
    count = 0

    for magnitude, rx_pos in scene.test_iter:
        magnitude = magnitude.to(device, non_blocking=True)
        rx_pos = rx_pos.to(device, non_blocking=True)

        ground_truth = magnitude.reshape(magnitude.shape[0], -1)
        predicted = model(rx_pos.reshape(-1, 3)).reshape(ground_truth.shape[0], -1)

        hits += int((predicted.argmax(dim=1) == ground_truth.argmax(dim=1)).sum().item())
        count += int(ground_truth.shape[0])

    model.train()

    return hits / max(count, 1)


def save_comparison_figure(
    ground_truth_map: np.ndarray,
    predicted_map: np.ndarray,
    figure_path: str,
    db_scale: bool,
) -> None:
    """Write one two-panel ground-truth vs predicted figure.

    ``db_scale`` switches between the linear layout ``train.py`` uses by default
    and its commented-out power-domain branch: ``10 * log10`` with a fixed
    ``[DB_VMIN, DB_VMAX]`` window, so both panels share one color scale and all
    figures of a run are comparable with each other.
    """
    if db_scale:
        ground_truth_map = 10.0 * np.log10(np.clip(ground_truth_map, 0.0, None) + DB_EPS)
        predicted_map = 10.0 * np.log10(np.clip(predicted_map, 0.0, None) + DB_EPS)
        image_kwargs = {"vmin": DB_VMIN, "vmax": DB_VMAX}
        titles = ("Ground Truth (dB)", "Predicted (dB)")
    else:
        image_kwargs = {}
        titles = ("Ground Truth", "Predicted")

    figure, axes = plt.subplots(2, 1, figsize=(8, 5), constrained_layout=True)

    for axis, title, data in zip(axes, titles, (ground_truth_map, predicted_map)):
        image = axis.imshow(
            data, aspect="equal", interpolation="nearest", **image_kwargs
        )
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_aspect("equal")

        divider = make_axes_locatable(axis)
        colorbar_axis = divider.append_axes("right", size="3.5%", pad=0.08)
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        colorbar.update_ticks()

    figure.savefig(figure_path, dpi=150)

    plt.close(figure)


def evaluate_and_save_random_test_samples(
    model: PositionMLP,
    scene,
    device: torch.device,
    output_dir: str,
) -> None:
    """Save ground-truth vs predicted figures for random test samples.

    Same sampling rule, batching and layout as
    ``train.evaluate_and_save_random_test_samples``; only the predictor differs
    (MLP forward pass instead of ``render_fast``).  The fixed seed means the
    figure index ``k`` here shows the same test location as figure ``k`` from a
    ``train.py`` run on this dataset.

    Every sample is written twice from the same forward pass: linear scale into
    ``pred_compare/`` and dB scale into ``pred_compare_db/``, under the same file
    name, so ``pred_compare/07.png`` and ``pred_compare_db/07.png`` are the two
    views of one test location.
    """
    linear_dir = os.path.join(output_dir, "pred_compare")
    db_dir = os.path.join(output_dir, "pred_compare_db")
    os.makedirs(linear_dir, exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)

    total_test_samples = len(scene.test_set)

    if total_test_samples == 0:
        print("[Evaluation] Test set is empty. Skipping figure generation.")
        return

    num_samples = min(NUM_EVAL_SAMPLES, total_test_samples)
    eval_batch_size = min(max(1, int(EVAL_BATCH_SIZE)), num_samples)

    # Fixed seed ensures that the same test samples are selected every run.
    random_generator = random.Random(SAMPLE_SEED)
    selected_indices = random_generator.sample(range(total_test_samples), num_samples)

    selected_samples = [scene.test_set[test_index] for test_index in selected_indices]
    ground_truth_batch_cpu = torch.stack(
        [
            magnitude.reshape(scene.beam_rows, scene.beam_cols)
            for magnitude, _ in selected_samples
        ],
        dim=0,
    )
    rx_pos_batch_cpu = torch.stack(
        [rx_pos.reshape(3) for _, rx_pos in selected_samples], dim=0
    )

    ground_truth_batch = ground_truth_batch_cpu.to(device, non_blocking=True)
    rx_pos_batch = rx_pos_batch_cpu.to(device, non_blocking=True)

    print(f"[Evaluation] Rendering {num_samples} random test samples with eval batch "
          f"size {eval_batch_size}...")

    model.eval()

    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        render_start = time.perf_counter()
        predicted_chunks = []

        for start_index in range(0, num_samples, eval_batch_size):
            end_index = min(start_index + eval_batch_size, num_samples)

            predicted_chunk = model(rx_pos_batch[start_index:end_index]).reshape(
                -1, scene.beam_rows, scene.beam_cols
            )

            predicted_chunks.append(predicted_chunk)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        render_time = time.perf_counter() - render_start

        predicted_batch = torch.cat(predicted_chunks, dim=0)

        if predicted_batch.shape[0] != num_samples:
            raise RuntimeError(
                "The number of rendered predictions does not match the requested "
                f"sample count: {predicted_batch.shape[0]} != {num_samples}"
            )

        ground_truth_normalized = normalize_mag_map(ground_truth_batch)

        predicted_batch_cpu = predicted_batch.detach().cpu()
        ground_truth_batch_cpu = ground_truth_normalized.detach().cpu()

    model.train()

    predicted_numpy = predicted_batch_cpu.numpy()
    ground_truth_numpy = ground_truth_batch_cpu.numpy()

    plot_and_save_start = time.perf_counter()

    for output_index in range(num_samples):
        ground_truth_map_numpy = ground_truth_numpy[output_index]
        predicted_map_numpy = predicted_numpy[output_index]
        figure_name = f"{output_index:02d}.png"

        save_comparison_figure(
            ground_truth_map_numpy,
            predicted_map_numpy,
            os.path.join(linear_dir, figure_name),
            db_scale=False,
        )
        save_comparison_figure(
            ground_truth_map_numpy,
            predicted_map_numpy,
            os.path.join(db_dir, figure_name),
            db_scale=True,
        )

    plot_and_save_time = time.perf_counter() - plot_and_save_start

    print(f"[Evaluation] Saved comparison figures to {linear_dir}")
    print(f"[Evaluation] Saved dB-scale comparison figures to {db_dir}")
    print(f"[Evaluation][Timing] render: {render_time:.4f} s")
    print(f"[Evaluation][Timing] plot_and_save: {plot_and_save_time:.4f} s")


def train(
    model_params,
    scene,
    device: torch.device,
    epochs: int,
    eval_every: int,
    output_dir: str,
    render_weights: str,
) -> Dict[str, object]:
    """Train the MLP, save the checkpoint and render the comparison figures."""
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    num_outputs = int(scene.beam_rows * scene.beam_cols)
    model = PositionMLP(num_outputs=num_outputs, hidden=HIDDEN, depth=DEPTH).to(device)

    parameters = count_parameters(model)
    iterations_per_epoch = len(scene.train_iter)
    total_iterations = iterations_per_epoch * epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_iterations), eta_min=FINAL_LR
    )

    print("-" * 96)
    print(f"[train_MLP_indoor] mlp_medium: hidden={HIDDEN} depth={DEPTH} "
          f"| params={parameters:,} | outputs={num_outputs}")
    print(f"[train_MLP_indoor] epochs={epochs} batch_size={scene.batch_size} "
          f"iters/epoch={iterations_per_epoch} total={total_iterations} "
          f"eval_every={eval_every}")

    trajectory: List[Dict[str, float]] = []
    best: Dict[str, object] = {"epoch": 0, "perloc_shape_db": float("inf")}
    best_state = None

    model.train()
    started = time.perf_counter()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_batches = 0

        for magnitude, rx_pos in scene.train_iter:
            magnitude = magnitude.to(device, non_blocking=True)
            rx_pos = rx_pos.to(device, non_blocking=True)

            ground_truth = magnitude.reshape(
                magnitude.shape[0], scene.beam_rows, scene.beam_cols
            )

            optimizer.zero_grad(set_to_none=True)
            predicted = model(rx_pos.reshape(-1, 3)).reshape(
                -1, scene.beam_rows, scene.beam_cols
            )
            loss = composite_magnitude_loss(
                predicted, ground_truth, topk_ratio=TOPK_RATIO, eps=LOSS_EPS
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.item())
            epoch_batches += 1

        is_last = epoch + 1 == epochs
        if not (is_last or (epoch + 1) % eval_every == 0):
            continue

        scale_db, shape_db, perloc_db, perloc_shape_db = evaluate_test_nmse(
            model, scene, device
        )
        top1 = evaluate_top1_accuracy(model, scene, device)

        trajectory.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss / max(epoch_batches, 1),
                "test_scale_nmse_db": scale_db,
                "test_shape_nmse_db": shape_db,
                "test_perloc_mean_db": perloc_db,
                "test_perloc_shape_mean_db": perloc_shape_db,
                "test_top1_accuracy": top1,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if perloc_shape_db < float(best["perloc_shape_db"]):
            best = {
                "epoch": epoch + 1,
                "perloc_shape_db": perloc_shape_db,
                "perloc_raw_db": perloc_db,
                "scale_nmse_db": scale_db,
                "shape_nmse_db": shape_db,
                "top1_accuracy": top1,
            }
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        elapsed = time.perf_counter() - started
        eta = elapsed / (epoch + 1) * (epochs - epoch - 1)
        print(
            f"  epoch {epoch + 1:>4}/{epochs} | train loss {epoch_loss / max(epoch_batches, 1):.6f} "
            f"| per-loc shape {perloc_shape_db:7.3f} dB (headline) "
            f"| per-loc raw {perloc_db:7.3f} dB "
            f"| mean-linear scale {scale_db:7.3f} / shape {shape_db:7.3f} dB "
            f"| top-1 {100.0 * top1:5.1f}% | eta {eta / 60.0:5.1f} min"
        )

    elapsed = time.perf_counter() - started

    os.makedirs(output_dir, exist_ok=True)

    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    checkpoint = {
        "name": "mlp_medium",
        "state_dict": final_state,
        "best_state_dict": best_state,
        "best": best,
        "config": {
            "hidden": HIDDEN,
            "depth": DEPTH,
            "num_outputs": num_outputs,
            "num_frequencies": 6,
            "include_input": True,
            "beam_rows": int(scene.beam_rows),
            "beam_cols": int(scene.beam_cols),
        },
        "training": {
            "epochs": epochs,
            "eval_every": eval_every,
            "batch_size": int(scene.batch_size),
            "iterations": total_iterations,
            "optimizer": "adam",
            "lr_init": BASE_LR,
            "lr_final": FINAL_LR,
            "lr_schedule": "cosine (per iteration)",
            "loss": "composite_magnitude_loss",
            "topk_ratio": TOPK_RATIO,
            "seed": SEED,
            "source_path": str(getattr(model_params, "source_path", "")),
            "beam_grid_mode": str(getattr(scene, "beam_grid_mode", "")),
            "train_samples": int(len(scene.train_set)),
            "test_samples": int(len(scene.test_set)),
            "render_weights": render_weights,
        },
        "parameters": parameters,
        "train_seconds": elapsed,
        "trajectory": trajectory,
    }
    torch.save(checkpoint, os.path.join(output_dir, "model.pth"))

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                k: v
                for k, v in checkpoint.items()
                if k not in ("state_dict", "best_state_dict")
            },
            handle,
            indent=2,
        )

    print(f"[train_MLP_indoor] Training done in {elapsed:.1f} s -> {output_dir}")

    # Figures are rendered from the requested weights; "best" falls back to the
    # final ones when no evaluation improved on the initial value.
    if render_weights == "best" and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[Evaluation] Rendering with the best epoch ({best['epoch']}).")
    else:
        print(f"[Evaluation] Rendering with the final epoch ({epochs}).")

    evaluate_and_save_random_test_samples(model, scene, device, output_dir)

    return {
        "name": "mlp_medium",
        "parameters": parameters,
        "output_dir": output_dir,
        "trajectory": trajectory,
        "best": best,
        "train_seconds": elapsed,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coordinate-MLP (mlp_medium) baseline on the indoor 63x63 dataset"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--source_path", type=str, default=DATASET_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval_every", type=int, default=EVAL_EVERY)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument(
        "--render_weights", type=str, default="final", choices=("final", "best"),
        help="Weights used for the pred_compare figures.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    repository_root = os.path.dirname(os.path.abspath(__file__))

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Resolve the data pipeline through the repo's own ModelParams so the split,
    # the loader and the beam-grid resolution match train.py exactly.
    defaults_parser = ArgumentParser()
    model_group = ModelParams(defaults_parser)
    namespace = defaults_parser.parse_args([])
    namespace.source_path = arguments.source_path
    namespace.model_path = ""
    namespace.batch_size = int(arguments.batch_size)
    namespace.num_epochs = int(arguments.epochs)
    model_params = model_group.extract(namespace)

    if not os.path.isdir(model_params.source_path):
        raise SystemExit(
            f"[train_MLP_indoor] Dataset directory '{model_params.source_path}' is missing."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = arguments.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(repository_root, output_dir)

    scene, _placeholder = build_scene(model_params)

    print("=" * 96)
    print("[train_MLP_indoor] Coordinate-MLP baseline (mlp_medium), indoor 63x63")
    print("=" * 96)
    print(f"  device      : {device}")
    print(f"  source_path : {model_params.source_path}")
    print(f"  train/test  : {len(scene.train_set)} / {len(scene.test_set)}")
    print(f"  beam grid   : {scene.beam_rows} x {scene.beam_cols} "
          f"(mode={scene.beam_grid_mode})")
    print(f"  epochs      : {arguments.epochs} | batch size {scene.batch_size} "
          f"| eval every {arguments.eval_every}")
    print(f"  figures     : {min(NUM_EVAL_SAMPLES, len(scene.test_set))} test samples "
          f"from the {arguments.render_weights} weights")
    print(f"  output      : {output_dir}")
    print("")

    result = train(
        model_params, scene, device,
        int(arguments.epochs), max(1, int(arguments.eval_every)),
        output_dir, str(arguments.render_weights),
    )

    final = result["trajectory"][-1]
    best = result["best"]

    print("")
    print("=" * 96)
    print("[train_MLP_indoor] SUMMARY (test set)")
    print("=" * 96)
    print(f"  parameters        : {result['parameters']:,}")
    print(f"  train seconds     : {result['train_seconds']:.1f}")
    print(f"  final epoch       : {int(final['epoch'])}")
    print(f"    per-loc shape   : {final['test_perloc_shape_mean_db']:.3f} dB")
    print(f"    per-loc raw     : {final['test_perloc_mean_db']:.3f} dB")
    print(f"    mean-linear     : scale {final['test_scale_nmse_db']:.3f} dB / "
          f"shape {final['test_shape_nmse_db']:.3f} dB")
    print(f"    top-1           : {100.0 * float(final['test_top1_accuracy']):.1f} %")
    print(f"  best epoch        : {int(best['epoch'])}")
    print(f"    per-loc shape   : {float(best['perloc_shape_db']):.3f} dB")
    print(f"    per-loc raw     : {float(best['perloc_raw_db']):.3f} dB")
    print(f"    mean-linear     : scale {float(best['scale_nmse_db']):.3f} dB / "
          f"shape {float(best['shape_nmse_db']):.3f} dB")
    print(f"    top-1           : {100.0 * float(best['top1_accuracy']):.1f} %")
    print("=" * 96)

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"[train_MLP_indoor] Summary written to {summary_path}")


if __name__ == "__main__":
    sys.exit(main())
