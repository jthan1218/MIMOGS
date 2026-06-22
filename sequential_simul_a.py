"""
sequential_simul_a.py

Replay / evaluation of the 11 training-progress checkpoints produced by
sequential_a.py (progress_000.pth ... progress_100.pth).

This script is EVALUATION / RENDERING ONLY:
    - It does NOT train.
    - It does NOT step or update any optimizer.
    - It does NOT call backward().
    - It does NOT generate or modify any checkpoint files.

For a fixed set of test sample(s) it renders the predicted beamspace
magnitude maps from every checkpoint (visualising how the model evolves over
training progress), and it evaluates rendering-fidelity / beam-selection
metrics versus training progress, saving plots, CSV and a JSON summary.

The checkpoints were saved with optimizer states stripped, so only the
GaussianModel parameters and the dynamic_gain_net weights are restored.

Run directly:
    python sequential_simul_a.py
"""

import os
import re
import csv
import json
import glob
import random
from types import SimpleNamespace
from argparse import ArgumentParser

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.loss import normalize_mag_map, magnitude_mse_loss  # noqa: F401 (magnitude_mse_loss kept for reuse)


SCRIPT_NAME = "sequential_simul_a"
LOG_TAG = "[SequentialSimulA]"
EPS = 1e-8


########################################################
# Small helpers
########################################################
def log(msg, quiet=False):
    if not quiet:
        print(f"{LOG_TAG} {msg}")


def to_py(obj):
    """Recursively convert numpy scalars/arrays to plain python types for JSON."""
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [to_py(v) for v in obj.tolist()]
    return obj


def ns_from_dict(d):
    """Convert a saved params dict to a SimpleNamespace (empty if None)."""
    if d is None:
        return SimpleNamespace()
    return SimpleNamespace(**dict(d))


def percent_from_path(path):
    """Extract the integer progress percent from a progress_XXX.pth filename."""
    name = os.path.basename(path)
    m = re.search(r"progress_(\d+)\.pth$", name)
    if m is None:
        return None
    return int(m.group(1))


def find_checkpoints(checkpoint_dir):
    """Return a list of (percent, path) sorted by percent for progress_*.pth."""
    pattern = os.path.join(checkpoint_dir, "progress_*.pth")
    paths = glob.glob(pattern)
    found = []
    for p in paths:
        pct = percent_from_path(p)
        if pct is None:
            continue
        found.append((pct, p))
    found.sort(key=lambda t: t[0])
    return found


########################################################
# Render / metrics primitives
########################################################
def make_render_kwargs(model_params):
    return dict(
        rx_shape=(2, 2),
        tx_shape=(4, 4),
        normalize_beam_weights=False,
        weight_floor=1e-4,
        max_active_rx_beams=getattr(model_params, "max_active_rx_beams", 2),
        max_active_tx_beams=getattr(model_params, "max_active_tx_beams", 2),
        renormalize_local_beam_weights=getattr(model_params, "renormalize_local_beam_weights", True),
    )


def render_sample(gaussians, scene, render_kwargs, tx_pos, device, idx):
    """Render one test sample. Returns (pred_mag, gt_mag) as 2D real tensors."""
    magnitude, rx_pos = scene.test_set[idx]
    rx_pos = rx_pos.to(device)
    gt_mag = magnitude.to(device).reshape(scene.beam_rows, scene.beam_cols)

    out = render(
        rx_pos=rx_pos,
        tx_pos=tx_pos,
        pc=gaussians,
        **render_kwargs,
    )
    pred_mag = out["render"]

    # Real-valued metrics require magnitudes: take abs() only if complex.
    if torch.is_complex(pred_mag):
        pred_mag = pred_mag.abs()
    if torch.is_complex(gt_mag):
        gt_mag = gt_mag.abs()

    pred_mag = pred_mag.reshape(scene.beam_rows, scene.beam_cols)
    return pred_mag, gt_mag


def compute_metrics(pred_mag, gt_mag):
    """Compute the per-sample metrics dict (python floats / ints)."""
    pred_n = normalize_mag_map(pred_mag, eps=EPS)
    gt_n = normalize_mag_map(gt_mag, eps=EPS)

    shape_mse = torch.mean((pred_n - gt_n) ** 2)
    raw_mse = torch.mean((pred_mag - gt_mag) ** 2)
    raw_nmse = torch.sum((pred_mag - gt_mag) ** 2) / (torch.sum(gt_mag ** 2) + EPS)
    raw_nmse_db = 10.0 * torch.log10(raw_nmse + EPS)

    top1_correct = int(
        torch.argmax(pred_mag.reshape(-1)).item()
        == torch.argmax(gt_mag.reshape(-1)).item()
    )

    return {
        "shape_mse": float(shape_mse.item()),
        "raw_mse": float(raw_mse.item()),
        "raw_nmse": float(raw_nmse.item()),
        "raw_nmse_db": float(raw_nmse_db.item()),
        "top1_correct": top1_correct,
    }


########################################################
# Plotting
########################################################
def save_map_png(path, mag_2d, title):
    """Save a single normalized beamspace map (vmin=0, vmax=1)."""
    vis = normalize_mag_map(mag_2d, eps=EPS).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 2.4))
    im = ax.imshow(vis, aspect="auto", origin="upper", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Tx beam")
    ax.set_ylabel("Rx beam")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_sequence_grid(path, gt_vis, percent_vis_list, sample_idx, metric_by_pct):
    """Combined 4x3 grid (12 cells), filled row-major:

        cells  1..11 : predicted maps at 0%, 10%, ..., 100% (in order)
        cell   12     : Ground Truth (last cell)

    All maps are peak-normalized and shown with vmin=0, vmax=1.
    """
    nrows, ncols = 4, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.8 * nrows))
    flat_axes = axes.reshape(-1)

    # Cells 1..11: predicted maps in ascending progress order.
    im = None
    for cell, (pct, vis) in enumerate(percent_vis_list):
        ax = flat_axes[cell]
        im = ax.imshow(vis, aspect="auto", origin="upper", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(f"{pct}%")
        ax.set_xticks([])
        ax.set_yticks([])

    # Final cell (index 11): Ground Truth.
    gt_ax = flat_axes[11]
    gt_ax.imshow(gt_vis, aspect="auto", origin="upper", vmin=0.0, vmax=1.0, cmap="viridis")
    gt_ax.set_title("GT")
    gt_ax.set_xticks([])
    gt_ax.set_yticks([])

    # Hide any leftover cells between the last prediction and the GT cell
    # (only relevant if fewer than 11 checkpoints were found).
    for cell in range(len(percent_vis_list), 11):
        flat_axes[cell].axis("off")

    fig.suptitle(f"Test sample {sample_idx:05d}: training-progress sequence")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_progress_plot(path, percents, values, ylabel, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(percents, values, marker="o", linestyle="-", color="tab:blue")
    ax.set_xlabel("Training progress (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


########################################################
# Main
########################################################
def run(args):
    quiet = args.quiet

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    if args.output_path:
        # Explicit override always wins.
        output_path = os.path.abspath(args.output_path)
    else:
        # Default: a "simul" folder that is a SIBLING of checkpoints/, i.e.
        #   outputs/sequential_a_.../checkpoints  ->  outputs/sequential_a_.../simul
        parent_dir = os.path.dirname(checkpoint_dir)
        output_path = os.path.join(parent_dir, "simul")
    os.makedirs(output_path, exist_ok=True)

    log(f"Checkpoint dir: {checkpoint_dir}", quiet)
    log(f"Output path (results saved here): {output_path}", quiet)

    # ----------------------------------------------------------------
    # Discover checkpoints
    # ----------------------------------------------------------------
    checkpoints = find_checkpoints(checkpoint_dir)
    if not checkpoints:
        raise FileNotFoundError(
            f"No 'progress_*.pth' checkpoints found in: {checkpoint_dir}"
        )

    found_percents = [pct for pct, _ in checkpoints]
    log(f"Found checkpoints: {found_percents}", quiet)

    expected = list(range(0, 101, 10))
    missing = [p for p in expected if p not in found_percents]
    if missing:
        log(f"WARNING: missing expected progress percents {missing}; "
            f"evaluating the {len(checkpoints)} found checkpoint(s).", quiet)

    # ----------------------------------------------------------------
    # Recover params from the FIRST checkpoint
    # ----------------------------------------------------------------
    first_ckpt = torch.load(checkpoints[0][1], map_location="cpu", weights_only=False)
    base_model_params = ns_from_dict(first_ckpt.get("model_params"))
    base_opt_params = ns_from_dict(first_ckpt.get("opt_params"))

    if args.source_path:
        base_model_params.source_path = args.source_path
    if args.data_device:
        base_model_params.data_device = args.data_device

    # Do NOT reuse the checkpoint's model_path as the eval output folder.
    base_model_params.model_path = output_path

    device = torch.device(
        getattr(base_model_params, "data_device", "cuda")
        if torch.cuda.is_available() else "cpu"
    )

    src = getattr(base_model_params, "source_path", "")
    if not src or not os.path.isdir(src):
        raise FileNotFoundError(
            "Dataset directory referenced by the checkpoint does not exist.\n"
            f"  source_path = {src!r}\n"
            "Pass --source_path to override it (must contain bs_info.yml, train.mat, test.mat)."
        )

    # ----------------------------------------------------------------
    # Build the Scene ONCE with a dummy/fresh GaussianModel.
    # Rendering always uses the per-checkpoint restored gaussians (passed
    # explicitly to render(pc=...)), so the dummy attached to the Scene is fine.
    # ----------------------------------------------------------------
    dummy_gaussians = GaussianModel(device=str(device))
    scene = Scene(base_model_params, dummy_gaussians)

    tx_pos = torch.tensor(scene.bs_position, dtype=torch.float32, device=device)
    test_size = len(scene.test_set)
    log(f"Test set size: {test_size}", quiet)

    # ----------------------------------------------------------------
    # Fixed sample selection (reproducible via render_seed)
    # ----------------------------------------------------------------
    rng = random.Random(args.render_seed)

    num_render = max(0, min(args.num_render_samples, test_size))
    render_indices = sorted(rng.sample(range(test_size), num_render)) if num_render > 0 else []

    eval_rng = random.Random(args.render_seed)
    if args.num_eval_samples and args.num_eval_samples > 0:
        n_eval = min(args.num_eval_samples, test_size)
        eval_indices = sorted(eval_rng.sample(range(test_size), n_eval))
    else:
        eval_indices = list(range(test_size))

    log(f"Eval samples: {len(eval_indices)}", quiet)
    log(f"Render sample indices: {render_indices}", quiet)

    # Persist selected indices
    with open(os.path.join(output_path, "selected_render_indices.txt"), "w", encoding="utf-8") as f:
        for idx in render_indices:
            f.write(f"{idx}\n")
    with open(os.path.join(output_path, "selected_eval_indices.txt"), "w", encoding="utf-8") as f:
        for idx in eval_indices:
            f.write(f"{idx}\n")

    render_root = os.path.join(output_path, "render_samples")
    os.makedirs(render_root, exist_ok=True)

    # ----------------------------------------------------------------
    # Ground-truth maps for the render samples (checkpoint-independent).
    # Saved once; also kept for the combined grid.
    # ----------------------------------------------------------------
    gt_vis_by_sample = {}
    for idx in render_indices:
        magnitude, _ = scene.test_set[idx]
        gt_mag = magnitude.to(device).reshape(scene.beam_rows, scene.beam_cols)
        if torch.is_complex(gt_mag):
            gt_mag = gt_mag.abs()
        sample_dir = os.path.join(render_root, f"sample_{idx:05d}")
        os.makedirs(sample_dir, exist_ok=True)
        save_map_png(
            os.path.join(sample_dir, "ground_truth.png"),
            gt_mag,
            f"Ground Truth (sample {idx:05d})",
        )
        gt_vis_by_sample[idx] = normalize_mag_map(gt_mag, eps=EPS).detach().cpu().numpy()

    # Collect predicted normalized maps + per-sample metric for the grid figures.
    # render_seq[sample_idx] = list of (percent, pred_vis_np)
    render_seq = {idx: [] for idx in render_indices}
    render_metric = {idx: {} for idx in render_indices}

    # ----------------------------------------------------------------
    # Evaluate each checkpoint
    # ----------------------------------------------------------------
    metrics_rows = []          # aggregated per checkpoint
    per_sample_rows = []       # per checkpoint x sample

    for pct, ckpt_path in checkpoints:
        log(f"Evaluating progress {pct:03d}% ...", quiet)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        mp = ns_from_dict(ckpt.get("model_params")) if ckpt.get("model_params") else base_model_params
        op = ns_from_dict(ckpt.get("opt_params")) if ckpt.get("opt_params") else base_opt_params
        if args.source_path:
            mp.source_path = args.source_path
        if args.data_device:
            mp.data_device = args.data_device

        render_kwargs = make_render_kwargs(mp)

        # Fresh model, restored (GaussianModel.restore calls training_setup; we
        # never step the optimizer, so this is purely a state-restore convenience).
        gaussians = GaussianModel(device=str(device))
        gaussians.restore(ckpt["gaussians"], op)
        gaussians.dynamic_gain_net.eval()

        sum_shape = sum_raw = sum_nmse = sum_nmse_db = 0.0
        sum_top1 = 0
        n = 0

        with torch.no_grad():
            # ---- evaluation samples ----
            for idx in eval_indices:
                pred_mag, gt_mag = render_sample(gaussians, scene, render_kwargs, tx_pos, device, idx)
                m = compute_metrics(pred_mag, gt_mag)

                per_sample_rows.append([
                    pct, idx,
                    m["shape_mse"], m["raw_mse"], m["raw_nmse"],
                    m["raw_nmse_db"], m["top1_correct"],
                ])

                sum_shape += m["shape_mse"]
                sum_raw += m["raw_mse"]
                sum_nmse += m["raw_nmse"]
                sum_nmse_db += m["raw_nmse_db"]
                sum_top1 += m["top1_correct"]
                n += 1

            # ---- render samples (visualisation) ----
            for idx in render_indices:
                pred_mag, gt_mag = render_sample(gaussians, scene, render_kwargs, tx_pos, device, idx)
                m = compute_metrics(pred_mag, gt_mag)
                render_metric[idx][pct] = m

                sample_dir = os.path.join(render_root, f"sample_{idx:05d}")
                save_map_png(
                    os.path.join(sample_dir, f"progress_{pct:03d}.png"),
                    pred_mag,
                    f"{pct}% (sample {idx:05d})\nnmse_db={m['raw_nmse_db']:.2f}",
                )
                pred_vis = normalize_mag_map(pred_mag, eps=EPS).detach().cpu().numpy()
                render_seq[idx].append((pct, pred_vis))

        mean_shape = sum_shape / max(n, 1)
        mean_raw = sum_raw / max(n, 1)
        mean_nmse = sum_nmse / max(n, 1)
        mean_nmse_db = sum_nmse_db / max(n, 1)
        top1_acc = sum_top1 / max(n, 1)

        metrics_rows.append({
            "progress_percent": pct,
            "checkpoint_path": ckpt_path,
            "num_eval_samples": n,
            "mean_shape_mse": mean_shape,
            "mean_raw_mse": mean_raw,
            "mean_raw_nmse": mean_nmse,
            "mean_raw_nmse_db": mean_nmse_db,
            "top1_accuracy": top1_acc,
        })

        log(f"progress {pct:03d}%: shape_mse={mean_shape:.6f}, "
            f"nmse_db={mean_nmse_db:.3f}, top1={top1_acc:.4f}", quiet)

        # Free memory; never keep all checkpoint models around.
        del gaussians
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Combined sequence grids
    # ----------------------------------------------------------------
    for idx in render_indices:
        sample_dir = os.path.join(render_root, f"sample_{idx:05d}")
        save_sequence_grid(
            os.path.join(sample_dir, "sequence_grid.png"),
            gt_vis_by_sample[idx],
            render_seq[idx],
            idx,
            render_metric[idx],
        )

    # ----------------------------------------------------------------
    # metrics.csv
    # ----------------------------------------------------------------
    metrics_csv = os.path.join(output_path, "metrics.csv")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "progress_percent", "checkpoint_path", "num_eval_samples",
            "mean_shape_mse", "mean_raw_mse", "mean_raw_nmse",
            "mean_raw_nmse_db", "top1_accuracy",
        ])
        for r in metrics_rows:
            writer.writerow([
                r["progress_percent"], r["checkpoint_path"], r["num_eval_samples"],
                r["mean_shape_mse"], r["mean_raw_mse"], r["mean_raw_nmse"],
                r["mean_raw_nmse_db"], r["top1_accuracy"],
            ])

    # ----------------------------------------------------------------
    # per_sample_metrics.csv
    # ----------------------------------------------------------------
    per_sample_csv = os.path.join(output_path, "per_sample_metrics.csv")
    with open(per_sample_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "progress_percent", "sample_idx", "shape_mse", "raw_mse",
            "raw_nmse", "raw_nmse_db", "top1_correct",
        ])
        writer.writerows(per_sample_rows)

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------
    percents = [r["progress_percent"] for r in metrics_rows]
    shape_vals = [r["mean_shape_mse"] for r in metrics_rows]
    nmse_db_vals = [r["mean_raw_nmse_db"] for r in metrics_rows]
    top1_vals = [r["top1_accuracy"] for r in metrics_rows]

    save_progress_plot(
        os.path.join(output_path, "progress_vs_shape_mse.png"),
        percents, shape_vals,
        "mean_shape_mse", "Shape MSE vs training progress",
    )
    save_progress_plot(
        os.path.join(output_path, "progress_vs_nmse_db.png"),
        percents, nmse_db_vals,
        "mean_raw_nmse_db", "Raw NMSE (dB) vs training progress",
    )
    save_progress_plot(
        os.path.join(output_path, "progress_vs_top1_accuracy.png"),
        percents, top1_vals,
        "top1_accuracy", "Top-1 accuracy vs training progress",
    )

    # ----------------------------------------------------------------
    # summary.json
    # ----------------------------------------------------------------
    def best_by(key, mode):
        if not metrics_rows:
            return None
        if mode == "min":
            r = min(metrics_rows, key=lambda x: x[key])
        else:
            r = max(metrics_rows, key=lambda x: x[key])
        return {"progress_percent": r["progress_percent"], key: r[key]}

    summary = {
        "checkpoint_dir": os.path.abspath(checkpoint_dir),
        "output_path": os.path.abspath(output_path),
        "evaluated_progress_percents": percents,
        "missing_progress_percents": missing,
        "test_set_size": test_size,
        "num_eval_samples": len(eval_indices),
        "selected_render_indices": render_indices,
        "selected_eval_indices": eval_indices,
        "best_by_shape_mse": best_by("mean_shape_mse", "min"),
        "best_by_raw_nmse_db": best_by("mean_raw_nmse_db", "min"),
        "best_by_top1_accuracy": best_by("top1_accuracy", "max"),
    }

    with open(os.path.join(output_path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_py(summary), f, indent=2)

    log(f"Wrote metrics to {metrics_csv}", quiet)
    log("Done.", quiet)


if __name__ == "__main__":
    parser = ArgumentParser(description="MIMOGS sequential_a checkpoint replay / evaluation")

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/sequential_a_20260622_075019/checkpoints",
    )
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--num_eval_samples", type=int, default=0)
    parser.add_argument("--num_render_samples", type=int, default=1)
    parser.add_argument("--render_seed", type=int, default=42)
    parser.add_argument("--data_device", type=str, default="")
    parser.add_argument("--source_path", type=str, default="")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    run(args)
