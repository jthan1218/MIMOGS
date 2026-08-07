"""E1 -- rendering fidelity evaluation for a trained MIMO-GS checkpoint.

Runs with zero arguments::

    python eval_render.py

The most recent run directory under ``outputs/`` that actually contains a
saved checkpoint is discovered automatically, the training-time configuration
is restored from it, the full test set is rendered with the same
``render_fast`` path that ``train.py`` uses, and per-location metrics plus
figures are written to ``analysis/<run_dir_name>/eval_render/``.

Normalization convention
------------------------
``train.py`` optimizes ``composite_magnitude_loss``, which uses BOTH
conventions with equal weight (0.4 / 0.4):

* scale term: RAW renderer output vs. max-normalized target,
* shape term: max-normalized renderer output vs. max-normalized target.

The convention is therefore genuinely ambiguous, so both NMSE variants are
reported per location and clearly labelled:

* ``NMSE_raw_dB``   -- ``pred`` is the raw renderer output (scale term).
* ``NMSE_shape_dB`` -- ``pred`` is max-normalized (shape term).

``NMSE_raw_dB`` is used as the headline metric for the figures because it is
the stricter of the two and matches ``evaluate_full_test_quality`` in
``train.py``. The target is always ``normalize_mag_map(target)``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import functools
import math
import os
import re
import sys
import time
from argparse import ArgumentParser, Namespace
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import scene.gaussian_model as gaussian_model_module
from arguments import ModelParams, OptimizationParams
from gaussian_renderer.fast_renderer import render_fast
from scene import GaussianModel, Scene
from utils.loss import normalize_mag_map


TOPK_VALUES = (1, 2, 4, 8, 16)
EPS = 1e-8
CHECKPOINT_NAME = "model.pth"
RUN_DIR_PATTERN = re.compile(r"^\d{8}_\d{6}$")


# ----------------------------------------------------------------------
# Checkpoint discovery / configuration restore
# ----------------------------------------------------------------------
def _run_dir_sort_key(run_dir: str) -> Tuple[int, float, str]:
    """Sort runs newest-first: timestamp-named dirs win, then mtime."""
    name = os.path.basename(os.path.normpath(run_dir))
    checkpoint_path = os.path.join(run_dir, CHECKPOINT_NAME)
    try:
        mtime = os.path.getmtime(checkpoint_path)
    except OSError:
        mtime = 0.0
    return (1 if RUN_DIR_PATTERN.match(name) else 0, mtime, name)


def discover_latest_run(outputs_root: str) -> str:
    """Return the most recent run directory that holds a saved checkpoint."""
    if not os.path.isdir(outputs_root):
        raise SystemExit(
            f"[eval_render] Output root '{outputs_root}' does not exist. "
            f"Train a model first, or pass --ckpt outputs/<run_dir>."
        )

    candidates = [
        os.path.join(outputs_root, name)
        for name in os.listdir(outputs_root)
        if os.path.isfile(os.path.join(outputs_root, name, CHECKPOINT_NAME))
    ]

    if not candidates:
        raise SystemExit(
            f"[eval_render] No run directory under '{outputs_root}' contains a "
            f"'{CHECKPOINT_NAME}'. Pass --ckpt outputs/<run_dir> explicitly."
        )

    candidates.sort(key=_run_dir_sort_key, reverse=True)
    return candidates[0]


def resolve_run_dir(ckpt_argument: Optional[str], outputs_root: str) -> Tuple[str, str]:
    """Return ``(run_dir, checkpoint_path)`` from an optional user argument."""
    if not ckpt_argument:
        run_dir = discover_latest_run(outputs_root)
        return run_dir, os.path.join(run_dir, CHECKPOINT_NAME)

    path = os.path.abspath(ckpt_argument)

    if os.path.isfile(path):
        return os.path.dirname(path), path

    checkpoint_path = os.path.join(path, CHECKPOINT_NAME)
    if not os.path.isfile(checkpoint_path):
        raise SystemExit(
            f"[eval_render] No checkpoint found at '{checkpoint_path}'. "
            f"Pass a run directory that contains '{CHECKPOINT_NAME}'."
        )
    return path, checkpoint_path


def _parse_scalar(text: str):
    """Best-effort literal parse of a value written by ``save_args``."""
    text = text.strip()
    if text in ("True", "False"):
        return text == "True"
    if text == "None":
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def read_run_args(run_dir: str) -> Dict[str, object]:
    """Parse ``run_args.txt`` written by ``train.save_args``."""
    path = os.path.join(run_dir, "run_args.txt")
    values: Dict[str, object] = {}

    if not os.path.isfile(path):
        return values

    section = ""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                continue
            if section == "Raw Args" or ":" not in line:
                # Raw Args duplicates the two typed groups; skip it so the
                # typed sections stay authoritative.
                continue
            key, _, raw_value = line.partition(":")
            values[key.strip()] = _parse_scalar(raw_value)

    return values


def read_cfg_args(run_dir: str) -> Dict[str, object]:
    """Read a ``cfg_args`` Namespace dump the way ``get_combined_args`` does."""
    path = os.path.join(run_dir, "cfg_args")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        namespace = eval(handle.read(), {"Namespace": Namespace}, {})
    return vars(namespace).copy()


def restore_config(run_dir: str, checkpoint: dict) -> Tuple[Namespace, Namespace]:
    """Rebuild the training-time model/optimization parameters.

    Priority (low to high): repo defaults from ``ModelParams`` /
    ``OptimizationParams`` -> ``cfg_args`` -> ``run_args.txt`` -> the
    ``model_params`` / ``opt_params`` dictionaries stored inside the
    checkpoint, which are the objects training actually ran with.
    """
    defaults_parser = ArgumentParser()
    model_group = ModelParams(defaults_parser)
    optimization_group = OptimizationParams(defaults_parser)
    defaults = vars(defaults_parser.parse_args([]))

    merged = dict(defaults)
    for source in (read_cfg_args(run_dir), read_run_args(run_dir)):
        merged.update({k: v for k, v in source.items() if v is not None})

    checkpoint_model = checkpoint.get("model_params", {}) or {}
    checkpoint_opt = checkpoint.get("opt_params", {}) or {}
    merged.update(checkpoint_model)
    merged.update(checkpoint_opt)

    merged["model_path"] = run_dir

    combined = Namespace(**merged)
    return model_group.extract(combined), optimization_group.extract(combined)


# ----------------------------------------------------------------------
# Checkpoint compatibility: gain-MLP width
# ----------------------------------------------------------------------
def gain_net_hidden_dim(checkpoint: dict) -> Optional[int]:
    """Recover the ``DynamicGainNet`` width this checkpoint was trained with.

    ``DynamicGainNet``'s default ``hidden_dim`` has changed in the repository
    since some checkpoints were written, so ``load_state_dict`` fails with a
    shape mismatch.  The width is unambiguously recoverable from the saved
    weights, and the Fourier-feature input dimension is verified against the
    live model so a genuinely incompatible checkpoint still fails loudly.

    Returns ``None`` when the checkpoint already matches the current default,
    i.e. when no override is needed.

    Defined here rather than in one of the eval_* scripts because those all
    import FROM this module; the reverse import would be circular.
    """
    gaussian_state = checkpoint.get("gaussians")
    if not isinstance(gaussian_state, (tuple, list)) or len(gaussian_state) < 13:
        return None

    net_state = gaussian_state[12]
    if not isinstance(net_state, dict) or "net.0.weight" not in net_state:
        return None

    first_layer = net_state["net.0.weight"]
    hidden_dim = int(first_layer.shape[0])
    input_dim = int(first_layer.shape[1])

    reference = gaussian_model_module.DynamicGainNet()
    expected_input = int(reference.net[0].weight.shape[1])
    if input_dim != expected_input:
        raise SystemExit(
            f"[eval_render] The checkpoint's gain MLP takes {input_dim} input "
            f"features but the current DynamicGainNet builds {expected_input}. "
            f"The Fourier-feature configuration changed since training; this "
            f"checkpoint cannot be evaluated with the current code."
        )

    if hidden_dim == int(reference.net[0].weight.shape[0]):
        return None
    return hidden_dim


@contextlib.contextmanager
def gain_net_width(hidden_dim: Optional[int]) -> Iterator[None]:
    """Temporarily build ``DynamicGainNet`` at a non-default hidden width."""
    if hidden_dim is None:
        yield
        return

    original = gaussian_model_module.DynamicGainNet
    gaussian_model_module.DynamicGainNet = functools.partial(
        original, hidden_dim=hidden_dim
    )
    try:
        yield
    finally:
        gaussian_model_module.DynamicGainNet = original


# ----------------------------------------------------------------------
# Model / scene construction
# ----------------------------------------------------------------------
def build_scene_and_model(
    model_params: Namespace,
    opt_params: Namespace,
    checkpoint: dict,
    device: torch.device,
) -> Tuple[Scene, GaussianModel]:
    """Mirror ``train.training`` construction, then restore the checkpoint."""
    gaussians = GaussianModel(
        target_gaussians=int(getattr(model_params, "target_gaussians", 25_000)),
        optimizer_type=getattr(opt_params, "optimizer_type", "default"),
        device=str(device),
        init_range=1.0,
        tie_covariance=bool(int(getattr(model_params, "tie_covariance", 0))),
    )

    scene = Scene(model_params, gaussians, shuffle=False)

    # ``restore`` runs ``training_setup``, which builds the LR schedules; the
    # exponential schedule divides by ``position_lr_max_steps``.
    opt_params.position_lr_max_steps = max(
        1, int(getattr(opt_params, "position_lr_max_steps", 0) or 0)
    )
    gaussians.restore(checkpoint["gaussians"], opt_params)
    gaussians.dynamic_gain_net.eval()

    return scene, gaussians


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def render_batch(
    rx_pos: torch.Tensor,
    tx_pos: torch.Tensor,
    gaussians: GaussianModel,
    scene: Scene,
    model_params: Namespace,
    use_cuda_rasterizer: bool,
) -> torch.Tensor:
    """Call ``render_fast`` exactly as ``train.py`` does and keep (B,Nr,Nt)."""
    rendered = render_fast(
        rx_pos=rx_pos.reshape(-1, 3),
        tx_pos=tx_pos,
        pc=gaussians,
        rx_shape=scene.rx_shape,
        tx_shape=scene.tx_shape,
        covariance_floor=1e-4,
        weight_floor=1e-4,
        max_active_rx_beams=int(getattr(model_params, "max_active_rx_beams", 2)),
        max_active_tx_beams=int(getattr(model_params, "max_active_tx_beams", 2)),
        use_cuda_rasterizer=use_cuda_rasterizer,
    )

    predicted = rendered["render"]
    if predicted.ndim == 2:
        predicted = predicted.unsqueeze(0)
    return predicted


def topk_metrics(
    predicted_flat: torch.Tensor,
    target_flat: torch.Tensor,
    k_values: Tuple[int, ...],
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Per-sample top-K overlap accuracy and power capture ratio.

    ``predicted_flat`` / ``target_flat`` are ``(B, Nr*Nt)`` magnitude maps.
    Ranking by magnitude and by power is identical for non-negative maps, but
    the captured *power* is accumulated on the squared magnitudes.
    """
    num_bins = target_flat.shape[1]
    target_power = target_flat.square()

    results: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    for k in k_values:
        k_eff = min(int(k), num_bins)

        gt_indices = torch.topk(target_flat, k=k_eff, dim=1, largest=True).indices
        pred_indices = torch.topk(predicted_flat, k=k_eff, dim=1, largest=True).indices

        gt_mask = torch.zeros_like(target_flat, dtype=torch.bool)
        gt_mask.scatter_(1, gt_indices, True)

        overlap = gt_mask.gather(1, pred_indices).sum(dim=1).float() / float(k_eff)

        genie_power = target_power.gather(1, gt_indices).sum(dim=1)
        selected_power = target_power.gather(1, pred_indices).sum(dim=1)
        capture = selected_power / genie_power.clamp_min(EPS)

        results[k] = (overlap, capture)

    return results


def evaluate_test_set(
    scene: Scene,
    gaussians: GaussianModel,
    model_params: Namespace,
    device: torch.device,
    batch_size: int,
    use_cuda_rasterizer: bool,
) -> Dict[str, np.ndarray]:
    """Render the whole test set and collect per-location metrics."""
    total_samples = len(scene.test_set)
    if total_samples == 0:
        raise SystemExit("[eval_render] The test set is empty; nothing to evaluate.")

    loader = DataLoader(
        scene.test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    tx_pos = torch.as_tensor(scene.bs_position, dtype=torch.float32, device=device)

    indices: List[int] = []
    nmse_raw: List[float] = []
    nmse_shape: List[float] = []
    positions: List[np.ndarray] = []
    topk_accumulator: Dict[int, List[float]] = {k: [] for k in TOPK_VALUES}
    capture_accumulator: Dict[int, List[float]] = {k: [] for k in TOPK_VALUES}

    skipped_zero_power = 0
    cursor = 0

    with torch.no_grad():
        for magnitude, rx_pos in loader:
            magnitude = magnitude.to(device, non_blocking=True)
            rx_pos = rx_pos.to(device, non_blocking=True)

            batch = magnitude.shape[0]
            batch_indices = torch.arange(cursor, cursor + batch)
            cursor += batch

            ground_truth = magnitude.reshape(batch, scene.beam_rows, scene.beam_cols)

            # Zero-power maps make the NMSE denominator degenerate; drop them
            # and report the count instead of silently clamping.
            peak = ground_truth.reshape(batch, -1).amax(dim=1)
            valid = peak > EPS
            num_valid = int(valid.sum().item())
            skipped_zero_power += batch - num_valid
            if num_valid == 0:
                continue

            predicted = render_batch(
                rx_pos, tx_pos, gaussians, scene, model_params, use_cuda_rasterizer
            )

            ground_truth = ground_truth[valid]
            predicted = predicted[valid]
            batch_indices = batch_indices[valid.cpu()]
            kept_positions = rx_pos.reshape(batch, 3)[valid]

            target_n = normalize_mag_map(ground_truth)
            predicted_n = normalize_mag_map(predicted)

            target_flat = target_n.reshape(num_valid, -1)
            predicted_flat = predicted.reshape(num_valid, -1)
            predicted_n_flat = predicted_n.reshape(num_valid, -1)

            target_energy = target_flat.square().sum(dim=1).clamp_min(EPS)
            raw_ratio = (predicted_flat - target_flat).square().sum(dim=1) / target_energy
            shape_ratio = (
                predicted_n_flat - target_flat
            ).square().sum(dim=1) / target_energy

            nmse_raw.extend(
                (10.0 * torch.log10(raw_ratio.clamp_min(1e-12))).cpu().tolist()
            )
            nmse_shape.extend(
                (10.0 * torch.log10(shape_ratio.clamp_min(1e-12))).cpu().tolist()
            )

            for k, (overlap, capture) in topk_metrics(
                predicted_flat, target_flat, TOPK_VALUES
            ).items():
                topk_accumulator[k].extend(overlap.cpu().tolist())
                capture_accumulator[k].extend(capture.cpu().tolist())

            indices.extend(batch_indices.tolist())
            positions.append(kept_positions.cpu().numpy())

    if not indices:
        raise SystemExit(
            "[eval_render] Every test map had zero power; no metric could be computed."
        )

    return {
        "index": np.asarray(indices, dtype=np.int64),
        "position": np.concatenate(positions, axis=0),
        "nmse_raw_db": np.asarray(nmse_raw, dtype=np.float64),
        "nmse_shape_db": np.asarray(nmse_shape, dtype=np.float64),
        "topk": {k: np.asarray(v, dtype=np.float64) for k, v in topk_accumulator.items()},
        "capture": {
            k: np.asarray(v, dtype=np.float64) for k, v in capture_accumulator.items()
        },
        "skipped_zero_power": skipped_zero_power,
    }


def summarize(values: np.ndarray) -> Dict[str, float]:
    percentiles = np.percentile(values, [5, 25, 75, 95])
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p5": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p75": float(percentiles[2]),
        "p95": float(percentiles[3]),
    }


def mean_linear_db(values_db: np.ndarray) -> float:
    """Average in the linear domain, then convert once (``train.py`` style)."""
    linear = np.power(10.0, values_db / 10.0)
    return float(10.0 * math.log10(max(float(np.mean(linear)), 1e-12)))


# ----------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------
def write_per_location_csv(path: str, results: Dict, scale_factor: float) -> None:
    header = ["index", "x", "y", "z", "NMSE_raw_dB", "NMSE_shape_dB"]
    for k in TOPK_VALUES:
        header.append(f"topk_acc_K{k}")
    for k in TOPK_VALUES:
        header.append(f"power_capture_K{k}")

    coordinates = results["position"] * scale_factor

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in range(results["index"].shape[0]):
            record = [
                int(results["index"][row]),
                f"{coordinates[row, 0]:.6f}",
                f"{coordinates[row, 1]:.6f}",
                f"{coordinates[row, 2]:.6f}",
                f"{results['nmse_raw_db'][row]:.6f}",
                f"{results['nmse_shape_db'][row]:.6f}",
            ]
            record += [f"{results['topk'][k][row]:.6f}" for k in TOPK_VALUES]
            record += [f"{results['capture'][k][row]:.6f}" for k in TOPK_VALUES]
            writer.writerow(record)


def write_summary_csv(path: str, summary: Dict[str, object]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(summary.keys()))
        writer.writerow(list(summary.values()))


def save_figure(figure, output_dir: str, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, f"{stem}.png"), dpi=200)
    figure.savefig(os.path.join(output_dir, f"{stem}.pdf"))
    plt.close(figure)


def select_qualitative_indices(nmse_db: np.ndarray) -> List[int]:
    """Two best, two median and two worst rows (positional, not dataset ids)."""
    order = np.argsort(nmse_db)
    count = order.shape[0]
    middle = count // 2

    picks = [order[0], order[min(1, count - 1)]]
    picks += [order[middle], order[min(middle + 1, count - 1)]]
    picks += [order[max(count - 2, 0)], order[count - 1]]

    unique: List[int] = []
    for pick in picks:
        if int(pick) not in unique:
            unique.append(int(pick))
    return unique


def plot_qualitative(
    output_dir: str,
    results: Dict,
    scene: Scene,
    gaussians: GaussianModel,
    model_params: Namespace,
    device: torch.device,
    use_cuda_rasterizer: bool,
) -> None:
    rows = select_qualitative_indices(results["nmse_raw_db"])
    labels = ["best", "best", "median", "median", "worst", "worst"][: len(rows)]

    dataset_indices = [int(results["index"][row]) for row in rows]
    magnitudes = torch.stack(
        [
            scene.test_set[i][0].reshape(scene.beam_rows, scene.beam_cols)
            for i in dataset_indices
        ],
        dim=0,
    ).to(device)
    rx_positions = torch.stack(
        [scene.test_set[i][1].reshape(3) for i in dataset_indices], dim=0
    ).to(device)
    tx_pos = torch.as_tensor(scene.bs_position, dtype=torch.float32, device=device)

    with torch.no_grad():
        predicted = render_batch(
            rx_positions, tx_pos, gaussians, scene, model_params, use_cuda_rasterizer
        )

    target_n = normalize_mag_map(magnitudes).cpu().numpy()
    predicted_np = predicted.cpu().numpy()

    figure, axes = plt.subplots(
        len(rows),
        2,
        figsize=(13, 2.3 * len(rows) + 1.2),
        squeeze=False,
        layout="constrained",
    )

    for panel, row in enumerate(rows):
        gt_map = target_n[panel]
        pred_map = predicted_np[panel]

        # One color scale per GT/prediction pair.
        vmin = 0.0
        vmax = float(max(gt_map.max(), pred_map.max(), EPS))

        gt_axis = axes[panel][0]
        pred_axis = axes[panel][1]

        gt_axis.imshow(
            gt_map, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax
        )
        image = pred_axis.imshow(
            pred_map, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax
        )

        gt_axis.set_title(
            f"GT ({labels[panel]}) idx={results['index'][row]}", fontsize=9
        )
        pred_axis.set_title(
            f"Rendered  NMSE = {results['nmse_raw_db'][row]:.2f} dB", fontsize=9
        )

        for axis in (gt_axis, pred_axis):
            axis.set_ylabel("Rx beam", fontsize=8)
            axis.tick_params(labelsize=7)
            if panel == len(rows) - 1:
                axis.set_xlabel("Tx beam", fontsize=8)

        figure.colorbar(image, ax=[gt_axis, pred_axis], fraction=0.02, pad=0.01)

    figure.suptitle(
        "Ground truth vs. rendered beam-pair maps (max-normalized target)", fontsize=11
    )
    # A shared colorbar spans each GT/prediction pair, so the constrained layout
    # requested at figure creation handles the spacing instead of tight_layout.
    figure.savefig(os.path.join(output_dir, "fig_qualitative.png"), dpi=200)
    figure.savefig(os.path.join(output_dir, "fig_qualitative.pdf"))
    plt.close(figure)


def plot_spatial_nmse(output_dir: str, results: Dict, scale_factor: float) -> None:
    coordinates = results["position"] * scale_factor

    figure, axis = plt.subplots(figsize=(7.2, 6.0))
    scatter = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=results["nmse_raw_db"],
        s=9,
        cmap="viridis",
    )
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("NMSE [dB]  (raw prediction vs. normalized target)")

    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title("Per-location rendering NMSE")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.25, linewidth=0.5)

    save_figure(figure, output_dir, "fig_spatial_nmse")


def plot_nmse_cdf(output_dir: str, results: Dict) -> None:
    figure, axis = plt.subplots(figsize=(6.6, 4.6))

    for values, label in (
        (results["nmse_raw_db"], "raw prediction (scale term)"),
        (results["nmse_shape_db"], "normalized prediction (shape term)"),
    ):
        ordered = np.sort(values)
        probabilities = np.arange(1, ordered.shape[0] + 1) / ordered.shape[0]
        axis.plot(ordered, probabilities, linewidth=1.6, label=label)

    axis.set_xlabel("NMSE [dB]")
    axis.set_ylabel("Empirical CDF")
    axis.set_title("Per-location NMSE distribution over the test set")
    axis.grid(alpha=0.3, linewidth=0.5)
    axis.legend(fontsize=8)
    axis.set_ylim(0.0, 1.0)

    save_figure(figure, output_dir, "fig_nmse_cdf")


def plot_topk_curves(output_dir: str, results: Dict) -> None:
    k_values = list(TOPK_VALUES)
    accuracy = [float(np.mean(results["topk"][k])) for k in k_values]
    capture = [float(np.mean(results["capture"][k])) for k in k_values]

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    axes[0].plot(k_values, accuracy, marker="o", linewidth=1.6)
    axes[0].set_title("Top-K beam-pair overlap accuracy")
    axes[0].set_ylabel("mean overlap fraction")

    axes[1].plot(k_values, capture, marker="s", color="tab:orange", linewidth=1.6)
    axes[1].set_title("Power capture ratio (predicted vs. genie selection)")
    axes[1].set_ylabel("mean captured power fraction")

    for axis, values in zip(axes, (accuracy, capture)):
        axis.set_xlabel("K")
        axis.set_xscale("log", base=2)
        axis.set_xticks(k_values)
        axis.set_xticklabels([str(k) for k in k_values])
        axis.grid(alpha=0.3, linewidth=0.5)
        axis.set_ylim(0.0, max(1.0, max(values) * 1.05))

    save_figure(figure, output_dir, "fig_topk_curves")


# ----------------------------------------------------------------------
# Cross-method summary table
# ----------------------------------------------------------------------
# The comparison table quotes the SHAPE convention (max-normalized prediction
# vs. max-normalized target).  It is the only one of the two conventions this
# script reports that is comparable across methods: the raw term scores a
# predictor against a target it was never rescaled to, so a method carrying a
# different absolute scale -- Sionna RT does -- is penalised for a
# normalization convention it has no way to know.  Both conventions stay in
# metrics_summary.csv untouched.
SUMMARY_NMSE_CONVENTION = "normalized prediction vs. normalized target (shape)"

COMPARISON_RT_RELATIVE = os.path.join("comparison_rt", "metrics_summary.csv")
MLP_COMPARE_RELATIVE = os.path.join("mlp_compare", "mlp_vs_gs.csv")
MLP_ROW_NAME = "mlp_small"

SUMMARY_COLUMNS = (
    "Method",
    "Mean NMSE [dB]",
    "Median NMSE [dB]",
    "Top-1 acc.",
    "Power capture @K=4",
    "Inference [ms/map]",
)
# The figure wraps the headers so each column stays narrow; the console table
# uses the flat labels above.
SUMMARY_COLUMN_LABELS = (
    "Method",
    "Mean NMSE\n[dB]",
    "Median NMSE\n[dB]",
    "Top-1\nacc.",
    "Power capture\n@ K=4",
    "Inference\n[ms/map]",
)
SUMMARY_COLUMN_WIDTHS = (0.22, 0.155, 0.165, 0.125, 0.175, 0.16)


def measure_inference_ms_per_map(
    scene: Scene,
    gaussians: GaussianModel,
    model_params: Namespace,
    device: torch.device,
    batch_size: int,
    use_cuda_rasterizer: bool,
    warmup_batches: int = 3,
    timed_batches: int = 20,
) -> float:
    """Milliseconds per rendered map, warmed up and CUDA-synchronised."""
    total = len(scene.test_set)
    if total == 0:
        return float("nan")

    count = min(batch_size, total)
    rx_pos = torch.stack(
        [scene.test_set[i][1].reshape(3) for i in range(count)], dim=0
    ).to(device)
    tx_pos = torch.as_tensor(scene.bs_position, dtype=torch.float32, device=device)

    with torch.no_grad():
        for _ in range(max(0, warmup_batches)):
            render_batch(
                rx_pos, tx_pos, gaussians, scene, model_params, use_cuda_rasterizer
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        started = time.perf_counter()
        for _ in range(max(1, timed_batches)):
            render_batch(
                rx_pos, tx_pos, gaussians, scene, model_params, use_cuda_rasterizer
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

    return 1000.0 * elapsed / float(max(1, timed_batches) * count)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    """Return a CSV's rows, or an empty list when the file is absent."""
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle)]


def _as_float(text: object) -> Optional[float]:
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def collect_summary_rows(
    analysis_root: str,
    run_name: str,
    results: Dict,
    inference_ms: float,
    notes: List[str],
) -> List[Dict[str, object]]:
    """Build the table rows, skipping any whose source numbers are missing."""
    rows: List[Dict[str, object]] = [
        {
            "Method": "MIMO-GS",
            "Mean NMSE [dB]": float(np.mean(results["nmse_shape_db"])),
            "Median NMSE [dB]": float(np.median(results["nmse_shape_db"])),
            "Top-1 acc.": float(np.mean(results["topk"][1])),
            "Power capture @K=4": float(np.mean(results["capture"][4])),
            "Inference [ms/map]": inference_ms,
            "_source": "this run",
            "_n": int(results["index"].shape[0]),
        }
    ]

    # -- Sionna RT, from the E2 comparison ------------------------------
    comparison_path = os.path.join(analysis_root, run_name, COMPARISON_RT_RELATIVE)
    sionna = next(
        (
            row
            for row in read_csv_rows(comparison_path)
            if row.get("predictor", "").strip().lower().startswith("sionna")
        ),
        None,
    )
    if sionna is None:
        notes.append(
            f"Sionna RT row skipped: no '{os.path.relpath(comparison_path)}'. "
            f"Run 'python eval_baseline_rt.py --allow_partial_match' to create it."
        )
    else:
        matched = _as_float(sionna.get("num_matched"))
        rows.append(
            {
                "Method": "Sionna RT*",
                "Mean NMSE [dB]": _as_float(sionna.get("NMSE_shape_mean_dB")),
                "Median NMSE [dB]": _as_float(sionna.get("NMSE_shape_median_dB")),
                "Top-1 acc.": _as_float(sionna.get("topk_acc_K1_mean")),
                "Power capture @K=4": _as_float(sionna.get("power_capture_K4_mean")),
                "Inference [ms/map]": "n/a (simulation)",
                "_source": os.path.relpath(comparison_path),
                "_n": int(matched) if matched else None,
            }
        )

    # -- MLP baseline, from the MLP-vs-GS sweep --------------------------
    mlp_path = os.path.join(analysis_root, MLP_COMPARE_RELATIVE)
    mlp = next(
        (
            row
            for row in read_csv_rows(mlp_path)
            if row.get("model", "").strip() == MLP_ROW_NAME
        ),
        None,
    )
    if mlp is None:
        notes.append(
            f"MLP ({MLP_ROW_NAME}) row skipped: no '{MLP_ROW_NAME}' entry in "
            f"'{os.path.relpath(mlp_path)}'. Run 'python eval_mlp_compare.py' "
            f"to create it."
        )
    else:
        evaluated = _as_float(mlp.get("num_evaluated"))
        rows.append(
            {
                "Method": f"MLP ({MLP_ROW_NAME})",
                "Mean NMSE [dB]": _as_float(mlp.get("nmse_shape_mean_dB")),
                "Median NMSE [dB]": _as_float(mlp.get("nmse_shape_median_dB")),
                "Top-1 acc.": _as_float(mlp.get("topk_acc_K1")),
                "Power capture @K=4": _as_float(mlp.get("power_capture_K4")),
                "Inference [ms/map]": _as_float(mlp.get("infer_ms_per_map")),
                "_source": os.path.relpath(mlp_path),
                "_n": int(evaluated) if evaluated else None,
            }
        )

    return rows


def format_summary_cell(column: str, value: object) -> str:
    """dB to 2 decimals, ratios to 3, strings passed through."""
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "n/a"
    if "NMSE" in column:
        return f"{float(value):.2f}"
    if "Inference" in column:
        return f"{float(value):.2f}"
    return f"{float(value):.3f}"


def build_summary_footnotes(rows: Sequence[Dict[str, object]], baseline_n: int) -> List[str]:
    """Footnote lines explaining conventions and any sample-set mismatch."""
    footnotes = [f"NMSE convention: {SUMMARY_NMSE_CONVENTION}, averaged per location."]
    for row in rows:
        if str(row["Method"]).endswith("*") and row.get("_n") not in (None, baseline_n):
            footnotes.append(
                f"* {row['Method'].rstrip('*')} is scored on the "
                f"{row['_n']} locations it could be matched to, not all "
                f"{baseline_n}; the two sample sets differ."
            )
    return footnotes


def write_summary_table_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(SUMMARY_COLUMNS) + ["num_locations", "source"])
        for row in rows:
            record = [
                row["Method"] if not isinstance(row[SUMMARY_COLUMNS[0]], float)
                else row[SUMMARY_COLUMNS[0]]
            ]
            for column in SUMMARY_COLUMNS[1:]:
                value = row.get(column)
                record.append("" if value is None else
                              (value if isinstance(value, str) else f"{float(value):.6f}"))
            record.append("" if row.get("_n") is None else int(row["_n"]))
            record.append(str(row.get("_source", "")))
            writer.writerow(record)


def print_summary_table(
    rows: Sequence[Dict[str, object]], footnotes: Sequence[str], caption: str
) -> None:
    """The same table the figure shows, as aligned console text."""
    cells = [
        [format_summary_cell(c, row.get(c)) if c != "Method" else str(row["Method"])
         for c in SUMMARY_COLUMNS]
        for row in rows
    ]
    widths = [
        max(len(SUMMARY_COLUMNS[i]), *(len(cell[i]) for cell in cells))
        for i in range(len(SUMMARY_COLUMNS))
    ]

    def line(values: Sequence[str]) -> str:
        parts = [values[0].ljust(widths[0])]
        parts += [values[i].rjust(widths[i]) for i in range(1, len(values))]
        return "  " + "  ".join(parts)

    print()
    print("=" * 78)
    print("[eval_render] SUMMARY COMPARISON")
    print("=" * 78)
    print(line(list(SUMMARY_COLUMNS)))
    print("  " + "-" * (sum(widths) + 2 * (len(widths) - 1)))
    for cell in cells:
        print(line(cell))
    print()
    print(f"  {caption}")
    for note in footnotes:
        print(f"  {note}")


def plot_summary_table(
    output_dir: str,
    rows: Sequence[Dict[str, object]],
    footnotes: Sequence[str],
    caption: str,
) -> None:
    """Render the comparison table as a compact figure.

    The table gets an explicit bbox that stops above the caption block, so the
    footnote lines can never overlap the last row however many of them there
    are.
    """
    body = [
        [format_summary_cell(c, row.get(c)) if c != "Method" else str(row["Method"])
         for c in SUMMARY_COLUMNS]
        for row in rows
    ]

    # Lay the figure out in inches first, then convert to axes fractions.
    header_height = 0.46
    row_height = 0.34
    text_height = 0.19 * (1 + len(footnotes)) + 0.10
    table_height = header_height + row_height * len(body)
    total_height = table_height + text_height

    figure, axis = plt.subplots(figsize=(7.6, total_height))
    axis.axis("off")

    table_bottom = text_height / total_height
    table = axis.table(
        cellText=body,
        colLabels=list(SUMMARY_COLUMN_LABELS),
        colWidths=list(SUMMARY_COLUMN_WIDTHS),
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, table_bottom, 1.0, 1.0 - table_bottom * 0.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("0.75")
        cell.set_linewidth(0.6)
        if row_index == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("0.90")
        elif str(rows[row_index - 1]["Method"]).startswith("MIMO-GS"):
            cell.set_facecolor("#eaf2fb")
        if column_index == 0 and row_index > 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.04

    line_step = 0.19 / total_height
    cursor = table_bottom - 0.055 / total_height
    for offset, line in enumerate([caption, *footnotes]):
        axis.text(
            0.0,
            cursor - offset * line_step,
            line,
            transform=axis.transAxes,
            fontsize=7.2,
            va="top",
            color="0.15" if offset == 0 else "0.35",
        )

    figure.savefig(
        os.path.join(output_dir, "fig_summary_table.png"),
        dpi=200,
        bbox_inches="tight",
    )
    figure.savefig(
        os.path.join(output_dir, "fig_summary_table.pdf"), bbox_inches="tight"
    )
    plt.close(figure)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MIMO-GS rendering fidelity evaluation (E1)"
    )
    parser.add_argument(
    "--ckpt",
    type=str,
    default="outputs/20260805_051724/model.pth",
    help="Run directory (outputs/<run_dir>) or a model.pth path.",
    )
    parser.add_argument("--outputs_root", type=str, default="outputs")
    parser.add_argument("--analysis_root", type=str, default="analysis")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Override the training-time batch size (0 keeps the restored value).",
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="",
        help="Override the dataset directory recorded in the checkpoint.",
    )
    return parser.parse_args()

def main() -> None:
    arguments = parse_arguments()
    repository_root = os.path.dirname(os.path.abspath(__file__))

    outputs_root = arguments.outputs_root
    if not os.path.isabs(outputs_root):
        outputs_root = os.path.join(repository_root, outputs_root)

    checkpoint_argument = arguments.ckpt
    if checkpoint_argument and not os.path.isabs(checkpoint_argument):
        # Resolve a relative --ckpt against the repository, so the default
        # works no matter which directory the script is launched from.
        relative_to_repo = os.path.join(repository_root, checkpoint_argument)
        if os.path.exists(relative_to_repo):
            checkpoint_argument = relative_to_repo

    run_dir, checkpoint_path = resolve_run_dir(checkpoint_argument, outputs_root)
    run_name = os.path.basename(os.path.normpath(run_dir))

    print("=" * 78)
    print(f"[eval_render] EVALUATING RUN : {run_name}")
    print(f"[eval_render] run directory  : {run_dir}")
    print(f"[eval_render] checkpoint     : {checkpoint_path}")
    print("=" * 78)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_params, opt_params = restore_config(run_dir, checkpoint)

    if arguments.source_path:
        model_params.source_path = os.path.abspath(arguments.source_path)
    if not os.path.isdir(getattr(model_params, "source_path", "")):
        raise SystemExit(
            f"[eval_render] Dataset directory '{getattr(model_params, 'source_path', '')}' "
            f"is missing. Pass --source_path <dir>."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda_rasterizer = bool(
        int(getattr(model_params, "use_cuda_rasterizer", 1))
    ) and device.type == "cuda"

    batch_size = int(arguments.batch_size) or int(getattr(model_params, "batch_size", 8))
    batch_size = max(1, batch_size)

    print(f"[eval_render] device={device} | batch_size={batch_size} | "
          f"cuda_rasterizer={int(use_cuda_rasterizer)}")
    print(f"[eval_render] source_path={model_params.source_path}")

    hidden_dim = gain_net_hidden_dim(checkpoint)

    if hidden_dim is not None:
        print(
            f"[eval_render] checkpoint DynamicGainNet hidden width: "
            f"{hidden_dim}"
        )

    with gain_net_width(hidden_dim):
        scene, gaussians = build_scene_and_model(
            model_params, opt_params, checkpoint, device
        )

    checkpoint_iteration = int(checkpoint.get("iteration", -1))
    num_gaussians = int(gaussians.get_xyz.shape[0])
    scale_factor = float(getattr(scene.test_set, "scale_factor", 1.0))

    print(f"[eval_render] test samples={len(scene.test_set)} | "
          f"beam grid = {scene.beam_rows} x {scene.beam_cols} | "
          f"gaussians={num_gaussians} | iteration={checkpoint_iteration}")

    results = evaluate_test_set(
        scene, gaussians, model_params, device, batch_size, use_cuda_rasterizer
    )

    output_dir = os.path.join(repository_root, arguments.analysis_root, run_name,
                              "eval_render")
    os.makedirs(output_dir, exist_ok=True)

    raw_stats = summarize(results["nmse_raw_db"])
    shape_stats = summarize(results["nmse_shape_db"])
    num_evaluated = int(results["index"].shape[0])

    write_per_location_csv(
        os.path.join(output_dir, "metrics_per_location.csv"), results, scale_factor
    )

    summary: Dict[str, object] = {
        "run_dir": run_name,
        "checkpoint_iteration": checkpoint_iteration,
        "num_epochs": int(getattr(model_params, "num_epochs", 0)),
        "target_gaussians": int(getattr(model_params, "target_gaussians", 0)),
        "num_gaussians_loaded": num_gaussians,
        "max_active_rx_beams": int(getattr(model_params, "max_active_rx_beams", 0)),
        "max_active_tx_beams": int(getattr(model_params, "max_active_tx_beams", 0)),
        "Nr": int(scene.beam_rows),
        "Nt": int(scene.beam_cols),
        "num_test_samples": int(len(scene.test_set)),
        "num_evaluated": num_evaluated,
        "num_skipped_zero_power": int(results["skipped_zero_power"]),
        "batch_size": batch_size,
        "device": str(device),
        "source_path": str(getattr(model_params, "source_path", "")),
        "position_scale_factor": scale_factor,
    }

    for prefix, stats, values in (
        ("NMSE_raw", raw_stats, results["nmse_raw_db"]),
        ("NMSE_shape", shape_stats, results["nmse_shape_db"]),
    ):
        summary[f"{prefix}_mean_dB"] = stats["mean"]
        summary[f"{prefix}_median_dB"] = stats["median"]
        summary[f"{prefix}_p5_dB"] = stats["p5"]
        summary[f"{prefix}_p25_dB"] = stats["p25"]
        summary[f"{prefix}_p75_dB"] = stats["p75"]
        summary[f"{prefix}_p95_dB"] = stats["p95"]
        # Linear-domain average converted once, matching train.evaluate_full_test_quality.
        summary[f"{prefix}_meanlinear_dB"] = mean_linear_db(values)

    for k in TOPK_VALUES:
        summary[f"topk_acc_K{k}_mean"] = float(np.mean(results["topk"][k]))
    for k in TOPK_VALUES:
        summary[f"power_capture_K{k}_mean"] = float(np.mean(results["capture"][k]))

    write_summary_csv(os.path.join(output_dir, "metrics_summary.csv"), summary)

    plot_qualitative(
        output_dir, results, scene, gaussians, model_params, device, use_cuda_rasterizer
    )
    plot_spatial_nmse(output_dir, results, scale_factor)
    plot_nmse_cdf(output_dir, results)
    plot_topk_curves(output_dir, results)

    # ------------------------------------------------------------------
    # Cross-method summary table (additive; nothing above is affected)
    # ------------------------------------------------------------------
    inference_ms = measure_inference_ms_per_map(
        scene, gaussians, model_params, device, batch_size, use_cuda_rasterizer
    )

    analysis_root = arguments.analysis_root
    if not os.path.isabs(analysis_root):
        analysis_root = os.path.join(repository_root, analysis_root)

    summary_notes: List[str] = []
    summary_rows = collect_summary_rows(
        analysis_root, run_name, results, inference_ms, summary_notes
    )
    for note in summary_notes:
        print(f"[eval_render] {note}")

    caption = (
        f"Checkpoint {run_name} (iteration {checkpoint_iteration}), "
        f"{num_evaluated} test locations, {scene.beam_rows}x{scene.beam_cols} "
        f"beam grid, {device.type}."
    )
    summary_footnotes = build_summary_footnotes(summary_rows, num_evaluated)

    write_summary_table_csv(
        os.path.join(output_dir, "summary_table.csv"), summary_rows
    )
    plot_summary_table(output_dir, summary_rows, summary_footnotes, caption)
    print_summary_table(summary_rows, summary_footnotes, caption)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print(f"[eval_render] SUMMARY -- run {run_name} (iteration {checkpoint_iteration})")
    print("=" * 78)
    print(f"  test locations evaluated : {num_evaluated} "
          f"(skipped zero-power: {results['skipped_zero_power']})")
    print(f"  gaussians                : {num_gaussians}")
    print()
    print(f"  {'NMSE [dB]':<34}{'mean':>9}{'median':>9}{'p5':>9}{'p95':>9}")
    for label, stats in (
        ("raw pred vs normalized target", raw_stats),
        ("normalized pred vs norm target", shape_stats),
    ):
        print(f"  {label:<34}{stats['mean']:>9.2f}{stats['median']:>9.2f}"
              f"{stats['p5']:>9.2f}{stats['p95']:>9.2f}")
    print()
    print(f"  {'K':>4}{'top-K overlap acc':>22}{'power capture':>18}")
    for k in TOPK_VALUES:
        print(f"  {k:>4}{np.mean(results['topk'][k]):>22.4f}"
              f"{np.mean(results['capture'][k]):>18.4f}")
    print()
    print(f"[eval_render] Outputs written to {output_dir}")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
