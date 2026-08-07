"""Training-density sweep: MIMO-GS vs the pure coordinate-MLP baseline.

Zero-argument runnable::

    python train_densitysweep.py

Motivation
----------
At full density (1 m interleaved grid, 15,787 training locations) the pure MLP
beats MIMO-GS -- -25.03 vs -23.19 dB shape NMSE, and on top-1 / power capture
too.  That regime is a dense *interpolation* problem, which favours a smooth
coordinate regressor.  The paper's premise is the opposite: measurements are
expensive and only sparse observations exist.  This sweep thins the training set
and measures how each model degrades.

Design
------
* Densities ``r in {1, .5, .25, .1, .05, .02, .01}`` -- random subsets of the
  TRAIN split only.  The TEST set is always the full original 3,947 locations.
* Both models are retrained at every density, 30 epochs each.  The epoch count is
  deliberately held FIXED rather than scaled with ``r`` so the comparison
  isolates *data quantity*; at small ``r`` this means far fewer gradient steps,
  which is the honest reading of "fewer measurements were taken".
* Scoring uses ``eval_render``'s conventions through ``eval_mlp_compare``'s
  generic predictor path, so every number here is directly comparable to
  ``analysis/mlp_compare/mlp_vs_gs.csv``.

Position-scale hazard
---------------------
``DeepMIMODataset`` normalizes positions by ``positions.abs().max()`` computed
*per file*.  A naive subsample can therefore shift the train-side scale factor
and silently place the model in a different coordinate frame from the test set.
Every subsample here force-includes the location attaining the global maximum
absolute coordinate, and the resulting scale factor is asserted equal to the
original.  See :func:`build_density_dataset`.

Outputs land in ``analysis/density_sweep/``; checkpoints in ``outputs/density/``.
Re-running skips any run whose checkpoint already exists (``--force_retrain``
overrides).
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from argparse import ArgumentParser
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import eval_render
from arguments import ModelParams
from eval_mlp_compare import (
    collect_row,
    evaluate_predictor,
    load_mlp,
    measure_inference_time,
)
from train_MLP import CONFIGS, build_scene, count_parameters, train_one


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATASET_NAME = "asu_campus_16by64_lt"
DENSITIES: Tuple[float, ...] = (1.00, 0.50, 0.25, 0.10, 0.05, 0.02, 0.01)
EPOCHS = 30
SEED = 0
BATCH_SIZE = 8
MLP_CONFIG = "mlp_small"
EXPECTED_BEAM_GRID = (16, 64)

# Known full-density reference numbers (mean shape NMSE, dB) and the tolerance
# the r=1.00 runs must land inside for the data plumbing to be trusted.
REFERENCE_R1 = {"MIMO-GS": -23.19, "MLP": -25.03}
REFERENCE_TOLERANCE_DB = 0.30

OUTPUT_ROOT = os.path.join("outputs", "density")
DATASET_ROOT = os.path.join(OUTPUT_ROOT, "_datasets")
ANALYSIS_DIR = os.path.join("analysis", "density_sweep")

# Categorical slots 1 and 2 of the reference data-viz palette.  Validated with
# the skill's six checks (ported to Python because node is unavailable here):
# light mode -- lightness band PASS, chroma PASS, CVD dE 24.7 (>=8) PASS,
# normal-vision dE 33.6 (>=15) PASS, contrast 4.30 / 3.12 (>=3) PASS.
COLOR_GS = "#2a78d6"   # blue
COLOR_MLP = "#eb6834"  # orange
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#d8d7d2"

MODEL_STYLE = {
    # colour is never the only channel: marker shape and dash pattern repeat it
    "MIMO-GS": {"color": COLOR_GS, "marker": "o", "linestyle": "-"},
    "MLP": {"color": COLOR_MLP, "marker": "s", "linestyle": "--"},
}


def repository_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


TIMING_FILE = "train_seconds.txt"


def record_train_seconds(run_dir: str, seconds: float) -> float:
    """Persist a run's training wall clock next to its checkpoint."""
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, TIMING_FILE), "w", encoding="utf-8") as handle:
        handle.write(f"{seconds:.3f}\n")
    return seconds


def read_train_seconds(run_dir: str) -> float:
    """Recover a skipped run's training wall clock, so resuming keeps the CSV whole."""
    path = os.path.join(run_dir, TIMING_FILE)
    if not os.path.isfile(path):
        return float("nan")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return float(handle.read().strip())
    except ValueError:
        return float("nan")


def density_tag(r: float) -> str:
    return f"{r:.2f}"


# ----------------------------------------------------------------------
# Dataset thinning
# ----------------------------------------------------------------------
def mean_nn_spacing(positions: np.ndarray) -> float:
    """Mean 2-D nearest-neighbour spacing in metres (z is constant here)."""
    from scipy.spatial import cKDTree

    if positions.shape[0] < 2:
        return float("nan")
    tree = cKDTree(positions[:, :2])
    distances, _ = tree.query(positions[:, :2], k=2)
    return float(np.mean(distances[:, 1]))


def build_density_dataset(
    r: float, source_dir: str, force: bool = False
) -> Tuple[str, int, float]:
    """Materialize a dataset directory whose train split is thinned to ``r``.

    Returns ``(directory, num_train_locations, mean_nn_spacing_m)``.

    The directory mimics the original layout: a subsampled ``train.mat``, a
    symlink to the untouched ``test.mat``, and a copy of ``bs_info.yml``.
    ``complex.mat`` is deliberately not carried over -- ``Scene`` never reads it
    and it is 1.6 GB.
    """
    target = os.path.join(DATASET_ROOT, f"{DATASET_NAME}_r{density_tag(r)}")
    os.makedirs(target, exist_ok=True)

    train_source = sio.loadmat(os.path.join(source_dir, "train.mat"))
    positions = train_source["positions"]
    magnitude = train_source["magnitude"]
    total = int(positions.shape[0])
    original_scale = float(np.abs(positions).max())

    count = max(2, int(round(r * total)))

    # Force-include the location attaining the global max |coordinate| so the
    # per-file normalization in DeepMIMODataset yields the same scale factor as
    # the untouched test split.  Without this the model can end up trained in a
    # different coordinate frame than it is evaluated in.
    anchor = int(np.argmax(np.abs(positions).max(axis=1)))
    rng = np.random.default_rng(SEED)
    pool = np.setdiff1d(np.arange(total), [anchor])
    chosen = rng.choice(pool, size=count - 1, replace=False)
    indices = np.sort(np.concatenate([[anchor], chosen]))

    subset_positions = positions[indices]
    subset_scale = float(np.abs(subset_positions).max())
    if not np.isclose(subset_scale, original_scale, rtol=0, atol=1e-3):
        raise SystemExit(
            f"[density] r={r}: subsample scale factor {subset_scale} != original "
            f"{original_scale}. Position normalization would diverge between "
            f"train and test. Refusing to continue."
        )

    spacing = mean_nn_spacing(subset_positions)
    train_path = os.path.join(target, "train.mat")

    if force or not os.path.isfile(train_path):
        sio.savemat(
            train_path,
            {"positions": subset_positions, "magnitude": magnitude[indices]},
            do_compression=False,
        )

    # test.mat: symlink to the original, so it can never drift.
    test_link = os.path.join(target, "test.mat")
    test_source = os.path.abspath(os.path.join(source_dir, "test.mat"))
    if os.path.islink(test_link) or os.path.isfile(test_link):
        if os.path.realpath(test_link) != test_source:
            os.remove(test_link)
    if not os.path.exists(test_link):
        os.symlink(test_source, test_link)

    bs_target = os.path.join(target, "bs_info.yml")
    if not os.path.isfile(bs_target):
        shutil.copy2(os.path.join(source_dir, "bs_info.yml"), bs_target)

    return target, int(indices.shape[0]), spacing


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train_gs(
    r: float, dataset_dir: str, run_dir: str, log_path: str, force: bool
) -> float:
    """Launch ``train.py`` as a subprocess.  Returns training wall clock (s)."""
    checkpoint = os.path.join(run_dir, eval_render.CHECKPOINT_NAME)
    if os.path.isfile(checkpoint) and not force:
        print(f"    [gs ] r={r:<5} checkpoint exists -> skipping training")
        return read_train_seconds(run_dir)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    command = [
        sys.executable, "train.py",
        "--source_path", dataset_dir,
        "--model_path", run_dir,
        "--num_epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--seed", str(SEED),
    ]
    started = time.perf_counter()
    # stdout/stderr go to a FILE, never a pipe: train.py emits a tqdm bar whose
    # volume would deadlock a PIPE-buffered child.
    with open(log_path, "w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, cwd=repository_root(), stdout=handle, stderr=subprocess.STDOUT
        )
    elapsed = time.perf_counter() - started

    if completed.returncode != 0 or not os.path.isfile(checkpoint):
        raise SystemExit(
            f"[density] train.py failed for r={r} (exit {completed.returncode}). "
            f"See {log_path}"
        )
    return record_train_seconds(run_dir, elapsed)


def train_mlp(r: float, dataset_dir: str, run_name: str, force: bool) -> float:
    """Train ``mlp_small`` on the thinned split, same recipe as train_MLP.py."""
    run_dir = os.path.join(repository_root(), OUTPUT_ROOT, run_name)
    if os.path.isfile(os.path.join(run_dir, "model.pth")) and not force:
        print(f"    [mlp] r={r:<5} checkpoint exists -> skipping training")
        return read_train_seconds(run_dir)

    defaults_parser = ArgumentParser()
    model_group = ModelParams(defaults_parser)
    namespace = defaults_parser.parse_args([])
    namespace.source_path = dataset_dir
    namespace.model_path = ""
    namespace.batch_size = BATCH_SIZE
    namespace.num_epochs = EPOCHS
    model_params = model_group.extract(namespace)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene, _ = build_scene(model_params)

    if (scene.beam_rows, scene.beam_cols) != EXPECTED_BEAM_GRID:
        raise SystemExit(
            f"[density] r={r}: MLP scene beam grid {(scene.beam_rows, scene.beam_cols)} "
            f"!= {EXPECTED_BEAM_GRID}."
        )

    started = time.perf_counter()
    train_one(
        run_name, CONFIGS[MLP_CONFIG], model_params, scene, device,
        EPOCHS, os.path.join(repository_root(), OUTPUT_ROOT),
    )
    return record_train_seconds(run_dir, time.perf_counter() - started)


# ----------------------------------------------------------------------
# Evaluation -- always on the FULL original test split
# ----------------------------------------------------------------------
def evaluate_gs(
    run_dir: str, canonical_source: str, device: torch.device
) -> Tuple[Dict[str, object], np.ndarray, int, float]:
    checkpoint = torch.load(
        os.path.join(run_dir, eval_render.CHECKPOINT_NAME),
        map_location="cpu", weights_only=False,
    )
    model_params, opt_params = eval_render.restore_config(run_dir, checkpoint)
    # Evaluate against the untouched dataset, not the thinned one it trained on.
    model_params.source_path = canonical_source

    hidden = eval_render.gain_net_hidden_dim(checkpoint)
    with eval_render.gain_net_width(hidden):
        scene, gaussians = eval_render.build_scene_and_model(
            model_params, opt_params, checkpoint, device
        )

    if (scene.beam_rows, scene.beam_cols) != EXPECTED_BEAM_GRID:
        raise SystemExit(f"[density] GS run {run_dir}: unexpected beam grid.")
    if os.path.realpath(scene.datadir) != os.path.realpath(canonical_source):
        raise SystemExit(f"[density] GS run {run_dir}: evaluated on the wrong dataset.")

    use_cuda = bool(int(getattr(model_params, "use_cuda_rasterizer", 1))) and (
        device.type == "cuda"
    )
    tx_pos = torch.as_tensor(scene.bs_position, dtype=torch.float32, device=device)

    def predict(rx_pos: torch.Tensor) -> torch.Tensor:
        return eval_render.render_batch(
            rx_pos, tx_pos, gaussians, scene, model_params, use_cuda
        )

    results = evaluate_predictor(predict, scene, device, BATCH_SIZE)
    inference_ms = measure_inference_time(predict, scene, device, BATCH_SIZE)
    parameters = count_parameters(gaussians.dynamic_gain_net) + int(
        sum(
            tensor.numel()
            for tensor in (
                gaussians._xyz, gaussians._xyz_tx, gaussians._scaling,
                gaussians._rotation, gaussians._scaling_tx, gaussians._rotation_tx,
                gaussians._opacity,
            )
        )
    )
    row = collect_row(results)
    index = results["index"]

    del gaussians, scene
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row, index, parameters, inference_ms


def evaluate_mlp(
    run_dir: str, canonical_scene, device: torch.device
) -> Tuple[Dict[str, object], np.ndarray, int, float]:
    model, checkpoint = load_mlp(run_dir, device)

    def predict(rx_pos: torch.Tensor) -> torch.Tensor:
        return model(rx_pos).reshape(
            -1, canonical_scene.beam_rows, canonical_scene.beam_cols
        )

    results = evaluate_predictor(predict, canonical_scene, device, BATCH_SIZE)
    inference_ms = measure_inference_time(predict, canonical_scene, device, BATCH_SIZE)
    parameters = int(checkpoint.get("parameters", count_parameters(model)))
    row = collect_row(results)
    index = results["index"]

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row, index, parameters, inference_ms


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def _style_axes(axis) -> None:
    axis.grid(True, which="major", color=GRID_COLOR, linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID_COLOR)
    axis.tick_params(colors=INK_SECONDARY, labelsize=9)


# A prediction whose shape NMSE is >= 0 dB carries no more information than the
# trivial constant predictor.  Both models collapse past a certain sparsity, and
# in that regime "who leads" is a comparison of two failures -- crossings there
# are noise, not a result.
DEGENERATE_NMSE_DB = 0.0


def find_crossovers(
    counts: Sequence[float],
    gs: Sequence[float],
    mlp: Sequence[float],
) -> List[Tuple[float, float, bool]]:
    """All sign changes of ``gs - mlp``, interpolated in log10(count) space.

    Returns ``(count, value, meaningful)`` per crossing, ordered sparse -> dense.
    ``meaningful`` is False when either model is already degenerate on both sides
    of the bracket.
    """
    log_counts = np.log10(np.asarray(counts, dtype=float))
    gs_array = np.asarray(gs, dtype=float)
    mlp_array = np.asarray(mlp, dtype=float)
    difference = gs_array - mlp_array

    crossings: List[Tuple[float, float, bool]] = []
    for i in range(len(difference) - 1):
        a, b = difference[i], difference[i + 1]
        if a == 0.0:
            t = 0.0
        elif a * b < 0.0:
            t = a / (a - b)
        else:
            continue
        log_x = log_counts[i] + t * (log_counts[i + 1] - log_counts[i])
        y = gs_array[i] + t * (gs_array[i + 1] - gs_array[i])
        degenerate = bool(
            max(gs_array[i], mlp_array[i]) >= DEGENERATE_NMSE_DB
            and max(gs_array[i + 1], mlp_array[i + 1]) >= DEGENERATE_NMSE_DB
        )
        crossings.append((float(10.0 ** log_x), float(y), not degenerate))
    return crossings


def headline_crossover(
    crossings: Sequence[Tuple[float, float, bool]]
) -> Optional[Tuple[float, float]]:
    """The densest crossing that is not inside the degenerate regime."""
    meaningful = [c for c in crossings if c[2]]
    if not meaningful:
        return None
    best = max(meaningful, key=lambda c: c[0])
    return best[0], best[1]


def plot_nmse_vs_density(
    records: List[Dict[str, object]], output_dir: str
) -> Tuple[Optional[Tuple[float, float]], List[Tuple[float, float, bool]]]:
    ordered = sorted({float(r["density"]) for r in records})
    counts, spacings = [], []
    series: Dict[str, List[float]] = {"MIMO-GS": [], "MLP": []}
    for r in ordered:
        for model in series:
            entry = next(
                x for x in records
                if float(x["density"]) == r and x["model"] == model
            )
            series[model].append(float(entry["nmse_shape_mean_dB"]))
        any_entry = next(x for x in records if float(x["density"]) == r)
        counts.append(float(any_entry["num_train_locations"]))
        spacings.append(float(any_entry["mean_spacing_m"]))

    figure, axis = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    _style_axes(axis)
    axis.set_xscale("log")

    for model, values in series.items():
        style = MODEL_STYLE[model]
        axis.plot(
            counts, values, label=model, color=style["color"],
            linestyle=style["linestyle"], linewidth=2.0,
            marker=style["marker"], markersize=8, markeredgecolor="white",
            markeredgewidth=1.2, zorder=3,
        )

    # Shade the regime where both models are no better than a constant predictor.
    degenerate_counts = [
        c for c, g, m in zip(counts, series["MIMO-GS"], series["MLP"])
        if max(g, m) >= DEGENERATE_NMSE_DB
    ]
    if degenerate_counts:
        band_hi = max(degenerate_counts) * 1.35
        axis.axvspan(
            min(counts) / 1.35, band_hi,
            color=GRID_COLOR, alpha=0.45, zorder=1, linewidth=0,
        )
        axis.text(
            np.sqrt(min(counts) / 1.35 * band_hi), 0.04,
            "both models degenerate\n($\\geq$ 0 dB: no better than\na constant map)",
            transform=axis.get_xaxis_transform(), fontsize=8,
            color=INK_SECONDARY, ha="center", va="bottom",
        )

    crossings = find_crossovers(counts, series["MIMO-GS"], series["MLP"])
    crossover = headline_crossover(crossings)
    if crossover is not None:
        x_cross, y_cross = crossover
        spacing_at_cross = float(
            np.interp(np.log10(x_cross), np.log10(counts), spacings)
        )
        axis.axvline(x_cross, color=INK_SECONDARY, linewidth=1.0,
                     linestyle=":", zorder=2)
        axis.annotate(
            f"crossover\n{x_cross:,.0f} locations\n(~{spacing_at_cross:.1f} m spacing)",
            xy=(x_cross, y_cross), xytext=(16, 10), textcoords="offset points",
            fontsize=9, color=INK_PRIMARY,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=GRID_COLOR, linewidth=0.8),
        )

    # Direct labels at the DENSE end, where the two curves are well separated;
    # at the sparse end they sit on top of each other.
    axis.set_xlim(min(counts) / 1.6, max(counts) * 2.9)
    for model, values in series.items():
        axis.annotate(
            model, xy=(counts[-1], values[-1]), xytext=(9, -3),
            textcoords="offset points", fontsize=9.5, ha="left", va="center",
            color=MODEL_STYLE[model]["color"], fontweight="bold",
        )

    # The dataset caption rides on the x-label so it can never collide with the
    # secondary spacing axis above or the tick labels below.
    axis.set_xlabel(
        "training locations  (log scale)\n"
        f"ASU campus 16x64 - full {int(max(counts)):,}-location grid thinned; "
        "test set fixed at 3,947 locations; 30 epochs throughout",
        color=INK_SECONDARY, fontsize=10,
    )
    axis.set_ylabel("mean shape NMSE [dB]   (lower is better)",
                    color=INK_SECONDARY, fontsize=10)
    axis.set_title(
        "Reconstruction accuracy vs training density",
        color=INK_PRIMARY, fontsize=13, loc="left", pad=34,
    )

    axis.set_xticks(counts)
    axis.set_xticklabels([f"{int(c):,}" for c in counts], rotation=0)
    axis.minorticks_off()

    secondary = axis.secondary_xaxis("top")
    secondary.set_xscale("log")
    secondary.set_xticks(counts)
    secondary.set_xticklabels([f"{s:.1f}" for s in spacings], fontsize=8.5)
    secondary.set_xlabel("mean inter-measurement spacing [m]",
                         color=INK_SECONDARY, fontsize=9.5, labelpad=2)
    secondary.tick_params(colors=INK_SECONDARY, labelsize=8.5)
    secondary.spines["top"].set_color(GRID_COLOR)
    secondary.minorticks_off()

    axis.legend(frameon=False, fontsize=10, labelcolor=INK_PRIMARY,
                loc="upper right", bbox_to_anchor=(0.995, 0.99))

    for extension in ("png", "pdf"):
        figure.savefig(os.path.join(output_dir, f"fig_nmse_vs_density.{extension}"),
                       dpi=200)
    plt.close(figure)
    return crossover, crossings


def plot_metrics_vs_density(records: List[Dict[str, object]], output_dir: str) -> Dict[str, Optional[float]]:
    panels = (
        ("topk_acc_K1", "top-1 beam accuracy"),
        ("topk_acc_K4", "top-4 overlap accuracy"),
        ("power_capture_K1", "power capture @ K=1"),
        ("power_capture_K4", "power capture @ K=4"),
    )
    ordered = sorted({float(r["density"]) for r in records})
    counts = []
    max_degenerate_count = 0.0
    for r in ordered:
        entry = next(x for x in records if float(x["density"]) == r)
        counts.append(float(entry["num_train_locations"]))
        both = [
            float(x["nmse_shape_mean_dB"]) for x in records
            if float(x["density"]) == r
        ]
        if max(both) >= DEGENERATE_NMSE_DB:
            max_degenerate_count = max(max_degenerate_count, float(entry["num_train_locations"]))

    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), constrained_layout=True)
    crossovers: Dict[str, Optional[float]] = {}

    for axis, (key, label) in zip(axes.ravel(), panels):
        _style_axes(axis)
        axis.set_xscale("log")
        values: Dict[str, List[float]] = {}
        for model in ("MIMO-GS", "MLP"):
            values[model] = [
                float(next(x for x in records
                           if float(x["density"]) == r and x["model"] == model)[key])
                for r in ordered
            ]
            style = MODEL_STYLE[model]
            axis.plot(
                counts, values[model], label=model, color=style["color"],
                linestyle=style["linestyle"], linewidth=2.0,
                marker=style["marker"], markersize=7, markeredgecolor="white",
                markeredgewidth=1.1, zorder=3,
            )
        # Higher is better here, so flip the sign for the shared crossover
        # finder.  Degeneracy is defined by the NMSE curves, not by these
        # accuracy values, so crossings inside the shaded band are discarded.
        crossings = find_crossovers(
            counts, [-v for v in values["MIMO-GS"]], [-v for v in values["MLP"]]
        )
        meaningful = [c for c in crossings if c[0] > max_degenerate_count]
        crossovers[key] = max((c[0] for c in meaningful), default=None)
        for crossing in meaningful:
            axis.axvline(crossing[0], color=INK_SECONDARY, linewidth=1.0,
                         linestyle=":", zorder=2)
        if max_degenerate_count > 0:
            axis.axvspan(min(counts) / 1.35, max_degenerate_count * 1.35,
                         color=GRID_COLOR, alpha=0.45, zorder=1, linewidth=0)

        axis.set_title(label, color=INK_PRIMARY, fontsize=11, loc="left")
        axis.set_xticks(counts)
        axis.set_xticklabels([f"{c/1000:.1f}k" if c >= 1000 else f"{int(c)}"
                              for c in counts], fontsize=8)
        axis.minorticks_off()
        axis.set_xlabel("training locations", color=INK_SECONDARY, fontsize=9)

    handles = [
        Line2D([0], [0], color=MODEL_STYLE[m]["color"],
               linestyle=MODEL_STYLE[m]["linestyle"], marker=MODEL_STYLE[m]["marker"],
               markersize=7, linewidth=2.0, label=m)
        for m in ("MIMO-GS", "MLP")
    ]
    figure.legend(handles=handles, frameon=False, fontsize=10,
                  labelcolor=INK_PRIMARY, loc="upper right", ncol=2)
    figure.suptitle(
        "Application metrics vs training density  (higher is better)",
        color=INK_PRIMARY, fontsize=12, x=0.01, ha="left",
    )

    for extension in ("png", "pdf"):
        figure.savefig(os.path.join(output_dir, f"fig_metrics_vs_density.{extension}"),
                       dpi=200)
    plt.close(figure)
    return crossovers


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
CSV_COLUMNS = [
    "model", "density", "num_train_locations", "mean_spacing_m", "train_iterations",
    "nmse_shape_mean_dB", "nmse_shape_median_dB",
    "nmse_raw_mean_dB", "nmse_raw_median_dB",
    "topk_acc_K1", "topk_acc_K4", "power_capture_K1", "power_capture_K4",
    "train_seconds", "parameters", "infer_ms_per_map", "num_evaluated", "run_dir",
]


def write_csv(records: List[Dict[str, object]], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in sorted(
            records, key=lambda r: (r["model"], -float(r["density"]))
        ):
            writer.writerow({k: record.get(k, "") for k in CSV_COLUMNS})


def print_table(records: List[Dict[str, object]]) -> None:
    print("")
    print("=" * 118)
    print("[density] SWEEP RESULTS  -- shape NMSE = normalized pred vs normalized "
          "target, per-location dB mean (eval_render headline)")
    print("=" * 118)
    header = (f"  {'model':<9}{'r':>6}{'train loc':>11}{'spacing':>9}"
              f"{'shape mean':>12}{'shape med':>11}{'raw mean':>10}"
              f"{'top-1':>8}{'top-4':>8}{'cap@1':>8}{'cap@4':>8}"
              f"{'train s':>10}{'ms/map':>9}")
    print(header)
    for model in ("MIMO-GS", "MLP"):
        for record in sorted(
            [r for r in records if r["model"] == model],
            key=lambda r: -float(r["density"]),
        ):
            train_seconds = record.get("train_seconds")
            seconds = ("  skipped" if train_seconds is None
                       or (isinstance(train_seconds, float) and np.isnan(train_seconds))
                       else f"{float(train_seconds):10.1f}")
            print(
                f"  {model:<9}{float(record['density']):>6.2f}"
                f"{int(record['num_train_locations']):>11,}"
                f"{float(record['mean_spacing_m']):>9.2f}"
                f"{float(record['nmse_shape_mean_dB']):>12.3f}"
                f"{float(record['nmse_shape_median_dB']):>11.3f}"
                f"{float(record['nmse_raw_mean_dB']):>10.3f}"
                f"{float(record['topk_acc_K1']):>8.4f}"
                f"{float(record['topk_acc_K4']):>8.4f}"
                f"{float(record['power_capture_K1']):>8.4f}"
                f"{float(record['power_capture_K4']):>8.4f}"
                f"{seconds}"
                f"{float(record['infer_ms_per_map']):>9.3f}"
            )
        print("")


def monotonicity_report(records: List[Dict[str, object]]) -> List[str]:
    """Flag any density step where NMSE improves as data is REMOVED."""
    notes: List[str] = []
    for model in ("MIMO-GS", "MLP"):
        subset = sorted(
            [r for r in records if r["model"] == model],
            key=lambda r: -float(r["density"]),
        )
        for previous, current in zip(subset, subset[1:]):
            delta = float(current["nmse_shape_mean_dB"]) - float(
                previous["nmse_shape_mean_dB"]
            )
            if delta < 0.0:
                notes.append(
                    f"{model}: r={float(previous['density']):.2f} -> "
                    f"{float(current['density']):.2f} IMPROVED by {-delta:.3f} dB "
                    f"while losing data (non-monotonic)"
                )
    return notes


def write_readme(
    records: List[Dict[str, object]],
    path: str,
    crossover: Optional[Tuple[float, float]],
    all_crossings: List[Tuple[float, float, bool]],
    metric_crossovers: Dict[str, Optional[float]],
    sanity: List[str],
    notes: List[str],
    total_seconds: float,
) -> None:
    lines = [
        "TRAINING-DENSITY SWEEP -- MIMO-GS vs pure coordinate-MLP",
        "=" * 70,
        "",
        "CONVENTIONS",
        "  shape NMSE (headline) : normalized prediction vs normalized target,",
        "                          averaged per location in dB -- identical to the",
        "                          eval_render.py summary table.",
        "  raw NMSE              : raw prediction vs normalized target, same averaging.",
        "  top-K / power capture : eval_render.topk_metrics, imported not reimplemented.",
        "  Every model is scored through eval_mlp_compare.evaluate_predictor, so these",
        "  numbers are directly comparable to analysis/mlp_compare/mlp_vs_gs.csv.",
        "",
        "SETUP",
        f"  dataset          : dataset/{DATASET_NAME} (16x64 beam grid)",
        f"  densities        : {', '.join(f'{d:.2f}' for d in DENSITIES)}",
        "  thinning         : random subset of the TRAIN split only; the TEST split is",
        "                     always the full original 3,947 locations, never subsampled.",
        f"  seed             : {SEED} (numpy default_rng for subsampling, torch for training)",
        f"  batch size       : {BATCH_SIZE}",
        f"  MLP config       : {MLP_CONFIG} "
        f"(hidden {CONFIGS[MLP_CONFIG]['hidden']}, depth {CONFIGS[MLP_CONFIG]['depth']})",
        "",
        "FIXED EPOCH COUNT -- a deliberate choice",
        f"  Both models train for {EPOCHS} epochs at EVERY density; the epoch count is NOT",
        "  scaled with r. An epoch is one pass over whatever data exists, so at r=0.01 a",
        "  run sees ~1/100 of the gradient steps of the r=1.00 run. That is the honest",
        "  reading of 'only sparse measurements were taken': the sparse regime is starved",
        "  of optimization budget as well as of data. Holding epochs fixed keeps the",
        "  comparison between the two MODELS clean at each density -- both are handed",
        "  exactly the same budget -- at the cost of confounding data quantity with step",
        "  count when reading ACROSS densities. Read each column as a head-to-head, and",
        "  the trend as 'degradation under a fixed training protocol'.",
        "",
        "POSITION-SCALE GUARD",
        "  DeepMIMODataset normalizes positions by positions.abs().max() computed per",
        "  file, so a naive train-split subsample can shift the train-side scale factor",
        "  and place the model in a different coordinate frame than the test split.",
        "  Every subsample force-includes the location attaining the global maximum",
        "  absolute coordinate, and the resulting scale factor is asserted equal to the",
        "  original (184.449) before training starts.",
        "",
        "SANITY CHECKS",
    ]
    lines.extend(f"  {line}" for line in sanity)
    lines.append("")
    lines.append("MONOTONICITY")
    if notes:
        lines.extend(f"  ! {line}" for line in notes)
    else:
        lines.append("  shape NMSE degrades monotonically with decreasing density for both models.")
    lines.append("")
    lines.append("CROSSOVER")
    lines.append("  A crossing inside the degenerate band (both models >= 0 dB, i.e. no")
    lines.append("  better than a constant map) compares two failures and is reported but")
    lines.append("  never headlined.")
    if crossover is None:
        best = min(records, key=lambda r: float(r["density"]))
        lines.append("  shape NMSE: no meaningful crossover over the swept range;")
        lines.append(f"  one model leads at every non-degenerate density down to "
                     f"r={float(best['density']):.2f}.")
    else:
        lines.append(f"  shape NMSE (headline): curves cross at ~{crossover[0]:,.0f} "
                     f"training locations ({crossover[1]:.2f} dB).")
    for count, value, meaningful in all_crossings:
        lines.append(
            f"    - crossing at ~{count:,.0f} locations ({value:.2f} dB)"
            + ("" if meaningful else "  [inside the degenerate band -- noise]")
        )
    for key, value in metric_crossovers.items():
        if value is not None:
            lines.append(f"  {key}: crosses at ~{value:,.0f} training locations.")
        else:
            lines.append(f"  {key}: no crossover over the swept range.")
    lines.append("")
    training_total = float(np.nansum([
        float(r.get("train_seconds", float("nan"))) for r in records
    ]))
    gs_total = float(np.nansum([
        float(r.get("train_seconds", float("nan")))
        for r in records if r["model"] == "MIMO-GS"
    ]))
    mlp_total = training_total - gs_total
    lines.append(f"TRAINING WALL CLOCK (sum over all 14 runs): "
                 f"{training_total / 60.0:.1f} min "
                 f"(MIMO-GS {gs_total / 60.0:.1f} min, MLP {mlp_total / 60.0:.1f} min)")
    lines.append(f"THIS PASS (incl. evaluation; training skipped when cached): "
                 f"{total_seconds / 60.0:.1f} min")
    lines.append("")
    lines.append("FILES")
    lines.append("  density_summary.csv         one row per (model, density)")
    lines.append("  fig_nmse_vs_density.png/pdf headline figure")
    lines.append("  fig_metrics_vs_density.*    2x2 application-metric panel")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training-density sweep: MIMO-GS vs pure coordinate-MLP"
    )
    parser.add_argument("--force_retrain", action="store_true",
                        help="Retrain even when a checkpoint already exists.")
    parser.add_argument("--densities", type=str, default="",
                        help="Comma-separated override of the density list.")
    parser.add_argument("--source_path", type=str,
                        default=os.path.join("dataset", DATASET_NAME))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    root = repository_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    canonical_source = arguments.source_path
    if not os.path.isabs(canonical_source):
        canonical_source = os.path.join(root, canonical_source)
    if os.path.basename(os.path.normpath(canonical_source)) != DATASET_NAME:
        raise SystemExit(
            f"[density] expected dataset variant '{DATASET_NAME}', got "
            f"'{canonical_source}'."
        )

    densities = DENSITIES
    if arguments.densities.strip():
        densities = tuple(float(x) for x in arguments.densities.split(",") if x.strip())

    for directory in (OUTPUT_ROOT, DATASET_ROOT, ANALYSIS_DIR):
        os.makedirs(os.path.join(root, directory), exist_ok=True)
    analysis_dir = os.path.join(root, ANALYSIS_DIR)

    print("=" * 118)
    print("[density] TRAINING-DENSITY SWEEP -- MIMO-GS vs pure coordinate-MLP")
    print("=" * 118)
    print(f"  device     : {device}")
    print(f"  dataset    : {canonical_source}")
    print(f"  densities  : {', '.join(f'{d:.2f}' for d in densities)}")
    print(f"  epochs     : {EPOCHS} (fixed at every density)  |  seed {SEED}")
    print("")

    sweep_started = time.perf_counter()
    records: List[Dict[str, object]] = []
    reference_index: Optional[np.ndarray] = None

    # One canonical scene, reused for every MLP evaluation.
    defaults_parser = ArgumentParser()
    model_group = ModelParams(defaults_parser)
    namespace = defaults_parser.parse_args([])
    namespace.source_path = canonical_source
    namespace.model_path = ""
    namespace.batch_size = BATCH_SIZE
    canonical_scene, _ = build_scene(model_group.extract(namespace))
    if (canonical_scene.beam_rows, canonical_scene.beam_cols) != EXPECTED_BEAM_GRID:
        raise SystemExit("[density] canonical scene has an unexpected beam grid.")
    print(f"  test set   : {len(canonical_scene.test_set)} locations, "
          f"beam grid {canonical_scene.beam_rows}x{canonical_scene.beam_cols}\n")

    for r in densities:
        tag = density_tag(r)
        print("-" * 118)
        dataset_dir, num_locations, spacing = build_density_dataset(
            r, canonical_source, force=arguments.force_retrain
        )
        print(f"[density] r={tag}  |  {num_locations:,} training locations  |  "
              f"mean spacing {spacing:.2f} m  |  {dataset_dir}")

        # --- MIMO-GS ---
        gs_run = os.path.join(root, OUTPUT_ROOT, f"gs_r{tag}")
        gs_log = os.path.join(root, OUTPUT_ROOT, "_logs", f"gs_r{tag}.log")
        started = time.perf_counter()
        gs_seconds = train_gs(r, dataset_dir, gs_run, gs_log, arguments.force_retrain)
        gs_row, gs_index, gs_parameters, gs_ms = evaluate_gs(
            gs_run, canonical_source, device
        )
        print(f"    [gs ] shape {gs_row['nmse_shape_mean_dB']:8.3f} dB | "
              f"raw {gs_row['nmse_raw_mean_dB']:8.3f} | "
              f"top-1 {gs_row['topk_acc_K1']:.4f} | "
              f"cap@4 {gs_row['power_capture_K4']:.4f} | "
              f"{time.perf_counter() - started:.0f} s elapsed")

        # --- MLP ---
        mlp_name = f"mlp_r{tag}"
        mlp_run = os.path.join(root, OUTPUT_ROOT, mlp_name)
        started = time.perf_counter()
        mlp_seconds = train_mlp(r, dataset_dir, mlp_name, arguments.force_retrain)
        mlp_row, mlp_index, mlp_parameters, mlp_ms = evaluate_mlp(
            mlp_run, canonical_scene, device
        )
        print(f"    [mlp] shape {mlp_row['nmse_shape_mean_dB']:8.3f} dB | "
              f"raw {mlp_row['nmse_raw_mean_dB']:8.3f} | "
              f"top-1 {mlp_row['topk_acc_K1']:.4f} | "
              f"cap@4 {mlp_row['power_capture_K4']:.4f} | "
              f"{time.perf_counter() - started:.0f} s elapsed")

        # --- identical test indices across every run ---
        for label, index in (("MIMO-GS", gs_index), ("MLP", mlp_index)):
            if reference_index is None:
                reference_index = index
            elif not np.array_equal(index, reference_index):
                raise SystemExit(
                    f"[density] r={tag} {label} was evaluated on different test "
                    "indices than an earlier run. Refusing to report."
                )

        for model, row, parameters, seconds, ms, run_dir in (
            ("MIMO-GS", gs_row, gs_parameters, gs_seconds, gs_ms, gs_run),
            ("MLP", mlp_row, mlp_parameters, mlp_seconds, mlp_ms, mlp_run),
        ):
            record: Dict[str, object] = {
                "model": model,
                "density": r,
                "num_train_locations": num_locations,
                "mean_spacing_m": spacing,
                # Epochs are fixed, so the gradient-step budget scales with the
                # data. Recorded because it is the confound when reading across
                # densities -- see the README.
                "train_iterations": (
                    -(-num_locations // BATCH_SIZE) * EPOCHS
                ),
                "train_seconds": seconds,
                "parameters": parameters,
                "infer_ms_per_map": ms,
                "run_dir": os.path.relpath(run_dir, root),
            }
            record.update(row)
            records.append(record)

    # --- sanity: r=1.00 must reproduce the known numbers ------------------
    sanity: List[str] = []
    for model, expected in REFERENCE_R1.items():
        match = [x for x in records
                 if x["model"] == model and float(x["density"]) == 1.00]
        if not match:
            sanity.append(f"{model} r=1.00 not run -- reference check skipped")
            continue
        observed = float(match[0]["nmse_shape_mean_dB"])
        delta = abs(observed - expected)
        state = "OK" if delta <= REFERENCE_TOLERANCE_DB else "MISMATCH"
        sanity.append(
            f"{model} r=1.00: {observed:.3f} dB vs known {expected:.2f} dB "
            f"(delta {delta:.3f} dB, tolerance {REFERENCE_TOLERANCE_DB:.2f}) -> {state}"
        )
    sanity.append(
        f"identical test indices across all {len(records)} runs "
        f"({0 if reference_index is None else reference_index.shape[0]} locations) -> OK"
    )

    print("=" * 118)
    print("[density] SANITY")
    for line in sanity:
        print(f"  {line}")

    notes = monotonicity_report(records)
    print("")
    print("[density] MONOTONICITY")
    if notes:
        for line in notes:
            print(f"  ! {line}")
    else:
        print("  shape NMSE degrades monotonically with density for both models.")

    print_table(records)

    write_csv(records, os.path.join(analysis_dir, "density_summary.csv"))
    crossover, all_crossings = plot_nmse_vs_density(records, analysis_dir)
    metric_crossovers = plot_metrics_vs_density(records, analysis_dir)
    total_seconds = time.perf_counter() - sweep_started
    write_readme(
        records, os.path.join(analysis_dir, "README.txt"),
        crossover, all_crossings, metric_crossovers, sanity, notes, total_seconds,
    )

    print("[density] CROSSOVER")
    if crossover is None:
        print("  shape NMSE: no meaningful crossover over the swept range.")
    else:
        print(f"  shape NMSE (headline): ~{crossover[0]:,.0f} training locations "
              f"({crossover[1]:.2f} dB)")
    for count, value, meaningful in all_crossings:
        print(f"    - crossing at ~{count:,.0f} locations ({value:.2f} dB)"
              + ("" if meaningful else "  [degenerate band -- noise]"))
    for key, value in metric_crossovers.items():
        print(f"  {key}: "
              + ("no crossover" if value is None else f"~{value:,.0f} locations"))

    print("")
    training_total = float(np.nansum([
        float(r.get("train_seconds", float("nan"))) for r in records
    ]))
    print(f"[density] training wall clock (all runs) {training_total / 60.0:.1f} min "
          f"| this pass {total_seconds / 60.0:.1f} min")
    print(f"[density] outputs -> {analysis_dir}")
    print("=" * 118)


if __name__ == "__main__":
    sys.exit(main())
