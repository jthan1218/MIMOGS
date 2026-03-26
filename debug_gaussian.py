#!/usr/bin/env python3
"""
Inspect per-Gaussian 4x16 beam contribution matrices using the latest MIMOGS git code path.

Default behavior:
- loads checkpoint from outputs/20260325_112050/model.pth
- reopens the dataset through Scene
- renders 50 random test samples with the SAME render settings as current eval path
- picks 5 Gaussians with the largest mean importance over those 50 renders
- for each selected Gaussian:
    * saves the raw 4x16 contribution matrix as text
    * saves an annotated heatmap image with each pixel value written on it
    * records rx_weights / tx_weights and the sample's pred/gt maps

Place this file in the repo root (same directory as train.py), then run:
    python debug_gaussian.py
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Repo-local imports. This script is meant to live next to train.py.
from gaussian_renderer import render
from scene import Scene, GaussianModel


DEFAULT_CHECKPOINT = "outputs/20260326_062415/model.pth"
DEFAULT_NUM_SAMPLES = 50
DEFAULT_NUM_GAUSSIANS = 5
DEFAULT_SEED = 12345


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect raw per-Gaussian 4x16 contribution matrices.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Checkpoint path. Default: outputs/20260325_112050/model.pth")
    parser.add_argument("--source_path", type=str, default=None,
                        help="Optional dataset root override. If omitted, uses checkpoint model_params[source_path].")
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES,
                        help="Number of random test samples to render.")
    parser.add_argument("--num_gaussians", type=int, default=DEFAULT_NUM_GAUSSIANS,
                        help="Number of Gaussians to inspect.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    return parser.parse_args()


def choose_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_namespace(obj: Any) -> SimpleNamespace:
    if isinstance(obj, SimpleNamespace):
        return obj
    if isinstance(obj, dict):
        return SimpleNamespace(**obj)
    # Fallback for argparse.Namespace-like objects already stored in ckpt
    return SimpleNamespace(**vars(obj))


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    required = ["gaussians", "model_params", "opt_params"]
    for key in required:
        if key not in ckpt:
            raise KeyError(f"Checkpoint missing key: {key}")
    return ckpt


def build_scene_and_model(
    ckpt: Dict[str, Any],
    checkpoint_path: str,
    source_path_override: str | None,
    device: torch.device,
):
    model_params = to_namespace(ckpt["model_params"])
    opt_params = to_namespace(ckpt["opt_params"])

    checkpoint_path = os.path.abspath(checkpoint_path)
    model_dir = os.path.dirname(checkpoint_path)

    # Keep behavior aligned with train.py / Scene expectations.
    model_params.model_path = model_dir
    if source_path_override is not None:
        model_params.source_path = os.path.abspath(source_path_override)
    elif getattr(model_params, "source_path", None):
        model_params.source_path = os.path.abspath(model_params.source_path)
    else:
        raise ValueError("source_path is missing in checkpoint. Provide --source_path.")

    gauss_blob = ckpt["gaussians"]
    if not isinstance(gauss_blob, (tuple, list)) or len(gauss_blob) < 4:
        raise ValueError("Unexpected checkpoint format for 'gaussians'.")

    target_gaussians = int(gauss_blob[0])
    optimizer_type = str(gauss_blob[1])
    init_range = float(gauss_blob[2])

    gaussians = GaussianModel(
        target_gaussians=target_gaussians,
        optimizer_type=optimizer_type,
        device=str(device),
        init_range=init_range,
    )
    gaussians.restore(gauss_blob, opt_params)
    # gaussians.eval()

    scene = Scene(model_params, gaussians)
    return scene, gaussians, model_params


@torch.no_grad()
def render_random_test_samples(
    scene,
    gaussians,
    num_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    total = len(scene.test_set)
    if total <= 0:
        raise ValueError("scene.test_set is empty.")

    num_samples = min(num_samples, total)
    rng = random.Random(seed)
    sample_indices = rng.sample(range(total), num_samples)

    tx_pos = torch.tensor(
        scene.bs_position,
        dtype=torch.float32,
        device=gaussians.get_xyz.device,
    )

    rows: List[Dict[str, Any]] = []
    for slot, dataset_idx in enumerate(sample_indices):
        magnitude, rx_pos = scene.test_set[dataset_idx]
        gt_mag = magnitude.to(gaussians.get_xyz.device).reshape(scene.beam_rows, scene.beam_cols)
        rx_pos = rx_pos.to(gaussians.get_xyz.device)

        # Match the current latest git eval behavior: no thresholding, no renormalization trick here.
        out = render(
            rx_pos=rx_pos,
            tx_pos=tx_pos,
            pc=gaussians,
            rx_shape=(2, 2),
            tx_shape=(4, 4),
            normalize_beam_weights=False,
            weight_floor=0.0,
        )

        rows.append(
            {
                "slot": slot,
                "dataset_idx": int(dataset_idx),
                "rx_pos": rx_pos.detach().cpu(),
                "gt_mag": gt_mag.detach().cpu(),
                "pred_mag": out["render"].detach().cpu(),
                "rx_weights": out["rx_weights"].detach().cpu(),
                "tx_weights": out["tx_weights"].detach().cpu(),
                "beam_contributions": out["beam_contributions"].detach().cpu(),  # (N,4,16)
                "importance": out["per_gaussian_importance"].detach().cpu(),
            }
        )
    return rows


def pick_gaussians(rendered_rows: List[Dict[str, Any]], num_gaussians: int):
    importance_stack = torch.stack([row["importance"].reshape(-1) for row in rendered_rows], dim=0)  # (S,N)
    mean_importance = importance_stack.mean(dim=0)
    max_importance, best_slots = importance_stack.max(dim=0)

    k = min(num_gaussians, mean_importance.numel())
    chosen = torch.topk(mean_importance, k=k, largest=True).indices.tolist()

    picked = []
    for g in chosen:
        picked.append(
            {
                "gaussian_idx": int(g),
                "mean_importance": float(mean_importance[g].item()),
                "max_importance": float(max_importance[g].item()),
                "best_slot": int(best_slots[g].item()),
            }
        )
    return picked


def matrix_to_text(arr: np.ndarray, decimals: int = 4) -> str:
    formatter = {"float_kind": lambda x: f"{x:.{decimals}f}"}
    return np.array2string(arr, formatter=formatter, max_line_width=200)


def annotate_matrix(ax, arr: np.ndarray, title: str, cmap: str = "viridis"):
    im = ax.imshow(arr, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Tx beam index")
    ax.set_ylabel("Rx beam index")
    ax.set_xticks(np.arange(arr.shape[1]))
    ax.set_yticks(np.arange(arr.shape[0]))

    vmax = float(np.max(arr)) if arr.size > 0 else 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            color = "white" if vmax > 0 and val > 0.5 * vmax else "black"
            ax.text(j, i, f"{val:.3g}", ha="center", va="center", color=color, fontsize=7)
    return im


def save_gaussian_artifacts(out_dir: Path, rendered_rows: List[Dict[str, Any]], picked: List[Dict[str, Any]]):
    summary_lines = []
    summary_lines.append("Per-Gaussian raw 4x16 contribution debug")
    summary_lines.append(f"num_rendered_samples: {len(rendered_rows)}")
    summary_lines.append("sample_dataset_indices: [" + ", ".join(str(r["dataset_idx"]) for r in rendered_rows) + "]")
    summary_lines.append("")

    for item in picked:
        g = item["gaussian_idx"]
        slot = item["best_slot"]
        row = rendered_rows[slot]

        contrib = row["beam_contributions"][g].numpy()           # (4,16)
        rx_w = row["rx_weights"][g].numpy()                      # (4,)
        tx_w = row["tx_weights"][g].numpy()                      # (16,)
        pred = row["pred_mag"].numpy()                           # (4,16)
        gt = row["gt_mag"].numpy()                               # (4,16)
        gt = gt / (np.max(gt) + 1e-8)

        stem = f"gaussian_{g:05d}_sampleSlot_{slot:02d}_datasetIdx_{row['dataset_idx']:05d}"

        # Text dump
        txt_lines = []
        txt_lines.append(f"gaussian_idx: {g}")
        txt_lines.append(f"sample_slot: {slot}")
        txt_lines.append(f"dataset_idx: {row['dataset_idx']}")
        txt_lines.append(f"mean_importance_over_{len(rendered_rows)}_samples: {item['mean_importance']:.8e}")
        txt_lines.append(f"max_importance_over_{len(rendered_rows)}_samples: {item['max_importance']:.8e}")
        txt_lines.append("")
        txt_lines.append("rx_weights (length 4):")
        txt_lines.append(matrix_to_text(rx_w[None, :], decimals=6))
        txt_lines.append("")
        txt_lines.append("tx_weights (length 16):")
        txt_lines.append(matrix_to_text(tx_w[None, :], decimals=6))
        txt_lines.append("")
        txt_lines.append("beam_contribution raw (4x16):")
        txt_lines.append(matrix_to_text(contrib, decimals=6))
        txt_lines.append("")
        txt_lines.append("pred_mag for the same sample (4x16):")
        txt_lines.append(matrix_to_text(pred, decimals=6))
        txt_lines.append("")
        txt_lines.append("gt_mag for the same sample (4x16):")
        txt_lines.append(matrix_to_text(gt, decimals=6))
        (out_dir / f"{stem}.txt").write_text("\n".join(txt_lines), encoding="utf-8")

        # NPY dump
        # np.save(out_dir / f"{stem}_contrib.npy", contrib)
        # np.save(out_dir / f"{stem}_rx_weights.npy", rx_w)
        # np.save(out_dir / f"{stem}_tx_weights.npy", tx_w)
        # np.save(out_dir / f"{stem}_pred.npy", pred)
        # np.save(out_dir / f"{stem}_gt.npy", gt)

        # Figure
        fig, axes = plt.subplots(3, 1, figsize=(18, 10), constrained_layout=True)
        im0 = annotate_matrix(axes[0], contrib, f"Gaussian {g} contribution matrix (4x16)")
        fig.colorbar(im0, ax=axes[0], fraction=0.025, pad=0.01)
        im1 = annotate_matrix(axes[1], pred, f"Predicted magnitude on sample dataset_idx={row['dataset_idx']}")
        fig.colorbar(im1, ax=axes[1], fraction=0.025, pad=0.01)
        im2 = annotate_matrix(axes[2], gt, f"GT magnitude (sample-wise max normalized) on sample dataset_idx={row['dataset_idx']}")
        fig.colorbar(im2, ax=axes[2], fraction=0.025, pad=0.01)
        fig.savefig(out_dir / f"{stem}.png", dpi=180)
        plt.close(fig)

        summary_lines.append(
            f"gaussian_idx={g}, best_sample_slot={slot}, dataset_idx={row['dataset_idx']}, "
            f"mean_importance={item['mean_importance']:.8e}, max_importance={item['max_importance']:.8e}"
        )
        summary_lines.append(f"  rx_weights: {np.array2string(rx_w, precision=6, suppress_small=False)}")
        summary_lines.append(f"  tx_weights: {np.array2string(tx_w, precision=6, suppress_small=False, max_line_width=200)}")
        summary_lines.append("  contribution_matrix:")
        summary_lines.append(matrix_to_text(contrib, decimals=6))
        summary_lines.append("")

    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (repo_root / checkpoint_path).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = choose_device(args.device)
    print(f"[Debug] device: {device}")
    print(f"[Debug] checkpoint: {checkpoint_path}")

    ckpt = load_checkpoint(str(checkpoint_path), device)
    scene, gaussians, model_params = build_scene_and_model(
        ckpt=ckpt,
        checkpoint_path=str(checkpoint_path),
        source_path_override=args.source_path,
        device=device,
    )

    print(f"[Debug] dataset source_path: {model_params.source_path}")
    print(f"[Debug] test samples available: {len(scene.test_set)}")
    print(f"[Debug] num gaussians in model: {gaussians.get_xyz.shape[0]}")
    print(f"[Debug] rendering {min(args.num_samples, len(scene.test_set))} random test samples...")

    rendered_rows = render_random_test_samples(
        scene=scene,
        gaussians=gaussians,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    picked = pick_gaussians(rendered_rows, args.num_gaussians)

    out_dir = checkpoint_path.parent / "gaussian_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_gaussian_artifacts(out_dir, rendered_rows, picked)

    print(f"[Debug] saved gaussian debug artifacts to: {out_dir}")
    print("[Debug] selected Gaussians:")
    for item in picked:
        print(
            f"  gaussian_idx={item['gaussian_idx']}, best_slot={item['best_slot']}, "
            f"mean_importance={item['mean_importance']:.6e}, max_importance={item['max_importance']:.6e}"
        )
    print(f"[Debug] summary: {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
