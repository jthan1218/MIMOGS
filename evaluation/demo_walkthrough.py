#!/usr/bin/env python3
"""
Create a mobility-style demo video for MIMOGS.

Run from the MIMOGS repository root after training, for example:

    python demo_walkthrough.py \
        --checkpoint outputs/20260430_120000/model.pth \
        --out outputs/20260430_120000/demo_walkthrough.mp4 \
        --path_mode test_order \
        --frames 240 \
        --fps 24

The video shows:
  - left: map/user trajectory in the same coordinate system used by the trained model
  - right: predicted 4 x 16 beamspace channel magnitude map at the current UE position

Notes:
  - This is a quasi-static spatial-channel demo. It visualizes H(rx_pos) changing with position.
  - Doppler is not injected into the renderer. The optional Doppler text overlay is only a rough
    kinematic indicator and is meaningful only if coordinates are in meters or --position_scale_m
    converts them to meters.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from gaussian_renderer import render
from scene import Scene, GaussianModel

C_LIGHT = 299_792_458.0


def _namespace_from_dict(d: dict | None) -> SimpleNamespace:
    return SimpleNamespace(**(d or {}))


def _ensure_opt_defaults(opt: SimpleNamespace) -> SimpleNamespace:
    """restore() calls training_setup(), so make sure required attrs exist."""
    defaults = {
        "iterations": 200_000,
        "position_lr_init": 0.0016,
        "position_lr_final": 0.000016,
        "position_lr_delay_mult": 0.01,
        "position_lr_max_steps": 200_000,
        "opacity_lr": 0.025,
        "opacity_lr_final": 0.003,
        "scaling_lr": 0.003,
        "rotation_lr": 0.0005,
        "optimizer_type": "default",
        "dynamic_gain_lr": 0.001,
        "dynamic_gain_lr_final": 0.0001,
    }
    for k, v in defaults.items():
        if not hasattr(opt, k):
            setattr(opt, k, v)
    return opt


def _strip_optimizer_states(model_args):
    """Keep learned tensors and dynamic gain net, drop optimizer states for inference."""
    model_args = list(model_args)
    if len(model_args) >= 14:
        model_args[11] = None  # Gaussian optimizer state
        model_args[13] = None  # dynamic-gain optimizer state
    return tuple(model_args)


def load_trained_model(checkpoint_path: str, source_path: str | None, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    model_params = _namespace_from_dict(ckpt.get("model_params", {}))
    opt_params = _ensure_opt_defaults(_namespace_from_dict(ckpt.get("opt_params", {})))

    if source_path:
        model_params.source_path = os.path.abspath(source_path)
    elif hasattr(model_params, "source_path"):
        model_params.source_path = os.path.abspath(model_params.source_path)
    else:
        raise ValueError("source_path is missing. Pass --source_path explicitly.")

    if not hasattr(model_params, "model_path") or not model_params.model_path:
        model_params.model_path = os.path.dirname(os.path.abspath(checkpoint_path))

    model_args = ckpt["gaussians"]
    gaussians = GaussianModel(
        target_gaussians=int(model_args[0]),
        optimizer_type=str(model_args[1]),
        device=str(device),
        init_range=float(model_args[2]),
    )
    gaussians.restore(_strip_optimizer_states(model_args), opt_params)
    gaussians.dynamic_gain_net.eval()

    scene = Scene(model_params, gaussians, shuffle=False)
    return scene, gaussians, model_params, opt_params, ckpt


def all_dataset_positions_and_magnitudes(scene: Scene) -> Tuple[np.ndarray, np.ndarray]:
    positions = []
    magnitudes = []
    for ds in (scene.train_set, scene.test_set):
        positions.append(ds.positions.detach().cpu().numpy())
        magnitudes.append(ds.magnitude.detach().cpu().numpy())
    return np.concatenate(positions, axis=0), np.concatenate(magnitudes, axis=0)


def build_trajectory(args, scene: Scene, all_pos: np.ndarray) -> np.ndarray:
    if args.path_csv:
        import pandas as pd

        df = pd.read_csv(args.path_csv)
        cols = [c for c in ["x", "y", "z"] if c in df.columns]
        if len(cols) == 3:
            traj = df[cols].to_numpy(dtype=np.float32)
        else:
            numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] < 3:
                raise ValueError("--path_csv must contain columns x,y,z or at least three numeric columns.")
            traj = numeric.iloc[:, :3].to_numpy(dtype=np.float32)

        if args.path_is_raw:
            scale = float(getattr(scene.train_set, "scale_factor", 1.0))
            traj = traj / max(scale, 1e-12)
        return resample_polyline(traj, args.frames)

    if args.path_mode == "test_order":
        pos = scene.test_set.positions.detach().cpu().numpy()
        idx = np.linspace(0, len(pos) - 1, args.frames).round().astype(int)
        return pos[idx].astype(np.float32)

    if args.path_mode == "line":
        p0 = all_pos[np.argmin(all_pos[:, 0])]
        p1 = all_pos[np.argmax(all_pos[:, 0])]
        t = np.linspace(0.0, 1.0, args.frames, dtype=np.float32)[:, None]
        return ((1.0 - t) * p0[None, :] + t * p1[None, :]).astype(np.float32)

    if args.path_mode == "loop":
        lo = all_pos.min(axis=0)
        hi = all_pos.max(axis=0)
        z = np.median(all_pos[:, 2])
        corners = np.array(
            [
                [lo[0], lo[1], z],
                [hi[0], lo[1], z],
                [hi[0], hi[1], z],
                [lo[0], hi[1], z],
                [lo[0], lo[1], z],
            ],
            dtype=np.float32,
        )
        return resample_polyline(corners, args.frames)

    raise ValueError(f"Unknown path_mode={args.path_mode}")


def resample_polyline(points: np.ndarray, frames: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == 1:
        return np.repeat(points, frames, axis=0)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-12:
        return np.repeat(points[:1], frames, axis=0)
    target = np.linspace(0, s[-1], frames)
    out = np.zeros((frames, 3), dtype=np.float32)
    for d in range(3):
        out[:, d] = np.interp(target, s, points[:, d])
    return out


def predict_channel_maps(
    traj: np.ndarray,
    tx_pos: Iterable[float],
    gaussians: GaussianModel,
    rx_shape=(2, 2),
    tx_shape=(4, 4),
) -> np.ndarray:
    device = gaussians.get_xyz.device
    dtype = gaussians.get_xyz.dtype
    tx = torch.tensor(tx_pos, dtype=dtype, device=device)
    maps = []
    with torch.no_grad():
        for p in traj:
            rx = torch.tensor(p, dtype=dtype, device=device)
            out = render(
                rx_pos=rx,
                tx_pos=tx,
                pc=gaussians,
                rx_shape=rx_shape,
                tx_shape=tx_shape,
                normalize_beam_weights=False,
                weight_floor=1e-4,
            )
            maps.append(out["render"].detach().float().cpu().numpy())
    return np.stack(maps, axis=0)


def compute_static_map_values(args, scene: Scene, gaussians: GaussianModel, map_pos: np.ndarray, map_mag: np.ndarray) -> np.ndarray:
    if args.map_source == "gt":
        return map_mag.reshape(map_mag.shape[0], -1).max(axis=1)

    # Predicted map source. This can be slower for large maps; use --map_max_points to subsample.
    device = gaussians.get_xyz.device
    dtype = gaussians.get_xyz.dtype
    tx = torch.tensor(scene.bs_position, dtype=dtype, device=device)
    vals = []
    with torch.no_grad():
        for p in map_pos:
            rx = torch.tensor(p, dtype=dtype, device=device)
            out = render(
                rx_pos=rx,
                tx_pos=tx,
                pc=gaussians,
                rx_shape=(2, 2),
                tx_shape=(4, 4),
                normalize_beam_weights=False,
                weight_floor=1e-4,
            )
            vals.append(float(out["render"].detach().float().max().cpu()))
    return np.asarray(vals, dtype=np.float32)


def to_display(x: np.ndarray, use_db: bool) -> np.ndarray:
    if use_db:
        return 20.0 * np.log10(np.maximum(x, 1e-12))
    return x


def estimate_doppler_overlay(args, traj: np.ndarray) -> np.ndarray | None:
    if args.carrier_ghz is None:
        return None
    if len(traj) < 2:
        return np.zeros(len(traj), dtype=np.float32)
    duration = args.duration_sec if args.duration_sec is not None else len(traj) / args.fps
    dt = max(duration / max(len(traj) - 1, 1), 1e-12)
    traj_m = traj * float(args.position_scale_m)
    speed = np.linalg.norm(np.diff(traj_m, axis=0), axis=1) / dt
    f_c = float(args.carrier_ghz) * 1e9
    f_d = speed * f_c / C_LIGHT
    return np.concatenate([[f_d[0] if len(f_d) else 0.0], f_d]).astype(np.float32)


def render_video(args, scene: Scene, gaussians: GaussianModel):
    all_pos, all_mag = all_dataset_positions_and_magnitudes(scene)

    # Subsample static map points for legible plotting and reasonable rendering time.
    if len(all_pos) > args.map_max_points:
        idx = np.linspace(0, len(all_pos) - 1, args.map_max_points).round().astype(int)
        map_pos = all_pos[idx]
        map_mag = all_mag[idx]
    else:
        map_pos = all_pos
        map_mag = all_mag

    traj = build_trajectory(args, scene, all_pos)
    channel_maps = predict_channel_maps(traj, scene.bs_position, gaussians)
    doppler_hz = estimate_doppler_overlay(args, traj)

    map_values = compute_static_map_values(args, scene, gaussians, map_pos, map_mag)
    map_disp = to_display(map_values, args.db)
    chan_disp = to_display(channel_maps, args.db)

    map_vmin, map_vmax = np.nanpercentile(map_disp, [2.0, 98.0])
    h_vmin, h_vmax = np.nanpercentile(chan_disp, [1.0, args.channel_percentile])
    if h_vmax <= h_vmin:
        h_vmax = h_vmin + 1e-6

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    try:
        for i, (p, h_raw, h_show) in enumerate(zip(traj, channel_maps, chan_disp)):
            fig = plt.figure(figsize=(12.0, 5.4), dpi=args.dpi)
            ax_map = fig.add_subplot(1, 2, 1)
            ax_h = fig.add_subplot(1, 2, 2)

            sc = ax_map.scatter(
                map_pos[:, 0],
                map_pos[:, 1],
                c=map_disp,
                s=args.map_point_size,
                alpha=0.70,
                vmin=map_vmin,
                vmax=map_vmax,
            )
            ax_map.plot(traj[:i+1, 0], traj[:i+1, 1], linewidth=1.5, alpha=0.85)
            ax_map.scatter([p[0]], [p[1]], s=90, marker="o", edgecolors="black", linewidths=1.2)

            bs = np.asarray(scene.bs_position, dtype=np.float32)
            if bs.shape[0] >= 2:
                ax_map.scatter([bs[0]], [bs[1]], s=120, marker="^", edgecolors="black", linewidths=1.2)

            ax_map.set_title(f"UE trajectory / frame {i + 1:03d}/{len(traj):03d}")
            ax_map.set_xlabel("x")
            ax_map.set_ylabel("y")
            ax_map.set_aspect("equal", adjustable="box")
            cb = fig.colorbar(sc, ax=ax_map, fraction=0.046, pad=0.04)
            cb.set_label(("max channel [dB]" if args.db else "max channel"))

            im = ax_h.imshow(h_show, aspect="auto", interpolation="nearest", vmin=h_vmin, vmax=h_vmax)
            best_flat = int(np.argmax(h_raw))
            best_rx, best_tx = np.unravel_index(best_flat, h_raw.shape)
            ax_h.scatter([best_tx], [best_rx], s=160, marker="s", facecolors="none", edgecolors="white", linewidths=2.0)
            ax_h.set_title("Predicted beamspace channel H(rx)")
            ax_h.set_xlabel("Tx beam index")
            ax_h.set_ylabel("Rx beam index")
            ax_h.set_xticks(range(h_raw.shape[1]))
            ax_h.set_yticks(range(h_raw.shape[0]))
            cb2 = fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
            cb2.set_label(("magnitude [dB]" if args.db else "magnitude"))

            info = f"best beam: Rx {best_rx}, Tx {best_tx}\nmax H: {float(np.max(h_raw)):.4g}\npos: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]"
            if doppler_hz is not None:
                info += f"\nrough max Doppler: {doppler_hz[i]:.1f} Hz"
            ax_h.text(
                0.02,
                0.98,
                info,
                transform=ax_h.transAxes,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.55),
                color="white",
                fontsize=9,
            )

            fig.tight_layout()
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
            rgb = rgba[:, :, :3]

            if writer is None:
                writer = cv2.VideoWriter(args.out, fourcc, args.fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer for {args.out}")
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            plt.close(fig)
    finally:
        if writer is not None:
            writer.release()

    print(f"[demo] saved video: {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Render a MIMOGS mobility demo video.")
    p.add_argument("--checkpoint", required=True, help="Path to trained model.pth")
    p.add_argument("--source_path", default=None, help="Dataset path. Defaults to checkpoint model_params.source_path")
    p.add_argument("--out", default="outputs/demo_walkthrough.mp4", help="Output MP4 path")
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("--path_mode", choices=["test_order", "line", "loop"], default="test_order")
    p.add_argument("--path_csv", default=None, help="Optional CSV with x,y,z trajectory points")
    p.add_argument("--path_is_raw", action="store_true", help="Divide CSV path by train_set.scale_factor before rendering")
    p.add_argument("--map_source", choices=["gt", "pred"], default="gt", help="Static map coloring source")
    p.add_argument("--map_max_points", type=int, default=5000)
    p.add_argument("--map_point_size", type=float, default=6.0)
    p.add_argument("--channel_percentile", type=float, default=99.0)
    p.add_argument("--db", action="store_true", help="Display channels as 20*log10(magnitude)")
    p.add_argument("--carrier_ghz", type=float, default=None, help="Optional rough Doppler overlay carrier in GHz")
    p.add_argument("--duration_sec", type=float, default=None, help="Physical duration of the trajectory for Doppler overlay")
    p.add_argument("--position_scale_m", type=float, default=1.0, help="Convert plotted/model position units to meters for Doppler overlay")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene, gaussians, *_ = load_trained_model(args.checkpoint, args.source_path, device)
    render_video(args, scene, gaussians)


if __name__ == "__main__":
    main()
