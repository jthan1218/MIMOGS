#!/usr/bin/env python3
import argparse
import os
import random
import sys
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from scipy.io import savemat

from arguments import ModelParams
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    def tqdm(x, **kwargs):
        return x


EPS = 1e-12


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_b_values(text: str) -> np.ndarray:
    vals = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))
    if not vals:
        raise ValueError("b_values must contain at least one integer.")
    arr = np.array(vals, dtype=np.int32)
    if np.any(arr <= 0):
        raise ValueError(f"All B values must be positive. Got: {arr.tolist()}")
    return arr


def _namespace_from_dict(d):
    return SimpleNamespace(**(d or {}))


def _ensure_opt_defaults(opt: SimpleNamespace) -> SimpleNamespace:
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
    args = list(model_args)
    if len(args) >= 13:
        args[11] = None
    if len(args) >= 14:
        args[13] = None
    return tuple(args)


def _collect_cli_overrides(parser: argparse.ArgumentParser) -> set:
    overrides = set()
    option_map = parser._option_string_actions
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("--"):
            if "=" in token:
                opt = token.split("=", 1)[0]
            else:
                opt = token
            action = option_map.get(opt)
            if action is not None and getattr(action, "dest", None):
                overrides.add(action.dest)
            i += 1
            continue
        i += 1
    return overrides


def _load_cfg_args(output_dir: str):
    cfg_path = os.path.join(output_dir, "cfg_args")
    if not os.path.exists(cfg_path):
        return Namespace(), cfg_path
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_text = f.read()
        cfg_ns = eval(cfg_text, {"Namespace": Namespace}, {})
        if not isinstance(cfg_ns, Namespace):
            raise TypeError(f"cfg_args did not evaluate to argparse.Namespace: {type(cfg_ns)}")
        return cfg_ns, cfg_path
    except Exception as exc:
        print(f"[Warning] Failed to parse cfg_args at {cfg_path}: {exc}")
        return Namespace(), cfg_path


def build_parser():
    parser = argparse.ArgumentParser(description="Compare MIMOGS vs baseline by BME metric.")
    model_params = ModelParams(parser)
    parser.set_defaults(source_path=None)

    parser.add_argument("--output_dir", type=str, default="outputs/20260512_082030")
    parser.add_argument("--checkpoint", type=str, default="model.pth")
    parser.add_argument("--mat_path", type=str, default="comparison.mat")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--num_test", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--b_values", type=str, default="1,2,4,6,8,12,16,24,32,48,64")
    parser.add_argument("--overhead_denominator", type=int, default=64)
    parser.add_argument("--random_trials", type=int, default=100)
    parser.add_argument("--disable_random_baseline", action="store_true")

    parser.add_argument("--rx_shape_h", type=int, default=2)
    parser.add_argument("--rx_shape_v", type=int, default=2)
    parser.add_argument("--tx_shape_h", type=int, default=4)
    parser.add_argument("--tx_shape_v", type=int, default=4)

    return parser, model_params


def merge_cfg_and_cli(parser: argparse.ArgumentParser):
    args_cmd = parser.parse_args()
    overrides = _collect_cli_overrides(parser)

    cfg_ns, cfg_path = _load_cfg_args(args_cmd.output_dir)
    merged = vars(args_cmd).copy()
    for k, v in vars(cfg_ns).items():
        if k not in overrides:
            merged[k] = v

    args = Namespace(**merged)
    args._cfg_path = cfg_path
    return args


def apply_required_defaults(args, model_defaults):
    if getattr(args, "source_path", None) in (None, ""):
        args.source_path = "./dataset/asu_campus_4by16_outdoor"

    if getattr(args, "max_active_rx_beams", None) is None:
        args.max_active_rx_beams = getattr(model_defaults, "max_active_rx_beams", 2)
    if getattr(args, "max_active_tx_beams", None) is None:
        args.max_active_tx_beams = getattr(model_defaults, "max_active_tx_beams", 2)
    if getattr(args, "renormalize_local_beam_weights", None) is None:
        args.renormalize_local_beam_weights = getattr(model_defaults, "renormalize_local_beam_weights", True)

    args.source_path = os.path.abspath(args.source_path)
    args.output_dir = os.path.abspath(args.output_dir)
    args.checkpoint_path = os.path.abspath(os.path.join(args.output_dir, args.checkpoint))
    args.mat_path = os.path.abspath(args.mat_path)
    return args


def build_scene_params(args):
    return SimpleNamespace(
        model_path=args.output_dir,
        source_path=args.source_path,
        data_device=args.device,
        eval=True,
        rx_num_beams=int(args.rx_num_beams),
        tx_num_beams=int(args.tx_num_beams),
        max_active_rx_beams=int(args.max_active_rx_beams),
        max_active_tx_beams=int(args.max_active_tx_beams),
        renormalize_local_beam_weights=bool(args.renormalize_local_beam_weights),
    )


def load_gaussians_from_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint must be dict, got {type(ckpt)}")
    if "gaussians" not in ckpt:
        raise KeyError("Checkpoint missing required key: 'gaussians'")

    model_args = ckpt["gaussians"]
    if not isinstance(model_args, (tuple, list)):
        raise TypeError(f"Checkpoint 'gaussians' must be tuple/list, got {type(model_args)}")
    if len(model_args) < 12:
        raise ValueError(f"Unexpected checkpoint 'gaussians' length={len(model_args)} (expected >=12).")

    opt_params = _ensure_opt_defaults(_namespace_from_dict(ckpt.get("opt_params", {})))
    gaussians = GaussianModel(
        target_gaussians=int(model_args[0]),
        optimizer_type=str(model_args[1]),
        device=str(device),
        init_range=float(model_args[2]),
    )

    # Backward compatibility:
    # - legacy checkpoints may store 12-tuple (without dynamic_gain_net states)
    # - current restore expects 14-tuple.
    model_args_list = list(model_args)
    if len(model_args_list) == 12:
        model_args_list.append(gaussians.dynamic_gain_net.state_dict())
        model_args_list.append(None)
    elif len(model_args_list) == 13:
        model_args_list.append(None)

    gaussians.restore(_strip_optimizer_states(tuple(model_args_list)), opt_params)
    if gaussians._opacity.dim() == 2 and gaussians._opacity.shape[1] != 1:
        print(
            f"[Warning] Restored opacity shape {tuple(gaussians._opacity.shape)} is not (N,1). "
            "Using first channel for evaluation compatibility."
        )
        gaussians._opacity = nn.Parameter(gaussians._opacity[:, :1].contiguous().requires_grad_(True))
    if hasattr(gaussians, "dynamic_gain_net") and gaussians.dynamic_gain_net is not None:
        gaussians.dynamic_gain_net.eval()
    return gaussians, ckpt


def format_checkpoint_error(checkpoint_path: str, exc: Exception):
    detail_lines = [
        f"[Error] Failed to load checkpoint: {checkpoint_path}",
        f"Type/Error: {type(exc).__name__}: {exc}",
        "Suggested fix: check that this is a MIMOGS model.pth produced by train.py/fine_tuning.py and matches current code.",
    ]
    return "\n".join(detail_lines)


def evaluate(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    B_values = parse_b_values(args.b_values)

    scene_params = build_scene_params(args)
    try:
        gaussians, _ = load_gaussians_from_checkpoint(args.checkpoint_path, device=device)
    except Exception as exc:
        raise RuntimeError(format_checkpoint_error(args.checkpoint_path, exc)) from exc

    scene = Scene(scene_params, gaussians, shuffle=False)
    beam_rows = int(scene.beam_rows)
    beam_cols = int(scene.beam_cols)
    total_beam_pairs = beam_rows * beam_cols

    if total_beam_pairs != 64 and int(args.overhead_denominator) == 64:
        print("[Warning] total beam pairs != 64 but overhead denominator is 64.")

    B_values = np.clip(B_values, 1, total_beam_pairs).astype(np.int32)
    B_values = np.unique(B_values)
    n_b = len(B_values)

    tx_pos = torch.tensor(
        scene.bs_position,
        dtype=torch.float32,
        device=gaussians.get_xyz.device,
    )

    n_test_total = len(scene.test_set)
    if args.num_test is None or args.num_test <= 0 or args.num_test > n_test_total:
        test_indices = np.arange(n_test_total, dtype=np.int64)
    else:
        test_indices = np.arange(args.num_test, dtype=np.int64)

    num_test_used = len(test_indices)
    if num_test_used == 0:
        raise ValueError("No test samples available for evaluation.")

    print(f"[Info] source_path: {args.source_path}")
    print(f"[Info] checkpoint_path: {args.checkpoint_path}")
    print(f"[Info] beam_rows x beam_cols: {beam_rows} x {beam_cols}")
    print(f"[Info] num_test_used: {num_test_used}")
    print(f"[Info] B_values: {B_values.tolist()}")
    print(f"[Info] cfg_args path: {args._cfg_path}")

    prior_power_sum = np.zeros(total_beam_pairs, dtype=np.float64)
    for i in tqdm(range(len(scene.train_set)), desc="Building baseline prior"):
        mag, _ = scene.train_set[i]
        mag_np = mag.detach().cpu().numpy().reshape(beam_rows, beam_cols)
        prior_power_sum += (mag_np ** 2).reshape(-1)
    prior_power = prior_power_sum / max(len(scene.train_set), 1)
    prior_rank = np.argsort(-prior_power)

    mimogs_align_sum = np.zeros(n_b, dtype=np.float64)
    mimogs_thr_sum = np.zeros(n_b, dtype=np.float64)
    baseline_align_sum = np.zeros(n_b, dtype=np.float64)
    baseline_thr_sum = np.zeros(n_b, dtype=np.float64)
    random_align_sum = np.zeros(n_b, dtype=np.float64)
    random_thr_sum = np.zeros(n_b, dtype=np.float64)

    run_random = not args.disable_random_baseline and int(args.random_trials) > 0
    rng = np.random.default_rng(args.seed)

    with torch.no_grad():
        for idx in tqdm(test_indices, desc="Evaluating test samples"):
            gt_mag, rx_pos = scene.test_set[int(idx)]
            gt_mag = gt_mag.to(gaussians.get_xyz.device).reshape(beam_rows, beam_cols)
            rx_pos = rx_pos.to(gaussians.get_xyz.device)
            gt_power = gt_mag ** 2
            gt_power_np = gt_power.detach().cpu().numpy().reshape(-1)

            true_best_idx = int(np.argmax(gt_power_np))
            true_best_power = float(np.max(gt_power_np))
            denom = max(true_best_power, EPS)

            out = render(
                rx_pos=rx_pos,
                tx_pos=tx_pos,
                pc=gaussians,
                rx_shape=(int(args.rx_shape_h), int(args.rx_shape_v)),
                tx_shape=(int(args.tx_shape_h), int(args.tx_shape_v)),
                normalize_beam_weights=False,
                weight_floor=1e-4,
                max_active_rx_beams=int(args.max_active_rx_beams),
                max_active_tx_beams=int(args.max_active_tx_beams),
                renormalize_local_beam_weights=bool(args.renormalize_local_beam_weights),
            )
            pred = out["render"]
            pred_np = np.abs(pred.detach().cpu().numpy())
            if pred_np.shape != (beam_rows, beam_cols):
                raise ValueError(
                    f"pred_mag shape mismatch: expected {(beam_rows, beam_cols)}, got {tuple(pred_np.shape)}"
                )
            pred_flat = pred_np.reshape(-1)
            pred_rank = np.argsort(-pred_flat)

            for bi, B in enumerate(B_values):
                B_int = int(B)

                # MIMOGS
                cand_m = pred_rank[:B_int]
                chosen_m = int(cand_m[np.argmax(gt_power_np[cand_m])])
                chosen_m_power = float(gt_power_np[chosen_m])
                align_m = 1.0 if chosen_m == true_best_idx else 0.0
                thr_m = np.clip(chosen_m_power / denom, 0.0, 1.0)
                mimogs_align_sum[bi] += align_m
                mimogs_thr_sum[bi] += thr_m

                # Statistical prior baseline
                cand_b = prior_rank[:B_int]
                chosen_b = int(cand_b[np.argmax(gt_power_np[cand_b])])
                chosen_b_power = float(gt_power_np[chosen_b])
                align_b = 1.0 if chosen_b == true_best_idx else 0.0
                thr_b = np.clip(chosen_b_power / denom, 0.0, 1.0)
                baseline_align_sum[bi] += align_b
                baseline_thr_sum[bi] += thr_b

                # Random baseline
                if run_random:
                    sampled = np.array(
                        [rng.choice(total_beam_pairs, size=B_int, replace=False) for _ in range(int(args.random_trials))]
                    )
                    sampled_powers = gt_power_np[sampled]
                    best_local = np.argmax(sampled_powers, axis=1)
                    chosen_idx = sampled[np.arange(sampled.shape[0]), best_local]
                    chosen_power = sampled_powers[np.arange(sampled.shape[0]), best_local]

                    align_r = np.mean(chosen_idx == true_best_idx)
                    thr_r = np.mean(np.clip(chosen_power / denom, 0.0, 1.0))
                    random_align_sum[bi] += float(align_r)
                    random_thr_sum[bi] += float(thr_r)

    overhead = B_values.astype(np.float64) / float(args.overhead_denominator)

    mimogs_alignment_accuracy = mimogs_align_sum / float(num_test_used)
    mimogs_throughput_ratio = mimogs_thr_sum / float(num_test_used)
    mimogs_BME = mimogs_alignment_accuracy * mimogs_throughput_ratio * (1.0 - overhead)

    baseline_alignment_accuracy = baseline_align_sum / float(num_test_used)
    baseline_throughput_ratio = baseline_thr_sum / float(num_test_used)
    baseline_BME = baseline_alignment_accuracy * baseline_throughput_ratio * (1.0 - overhead)

    if run_random:
        random_alignment_accuracy = random_align_sum / float(num_test_used)
        random_throughput_ratio = random_thr_sum / float(num_test_used)
        random_BME = random_alignment_accuracy * random_throughput_ratio * (1.0 - overhead)
    else:
        random_alignment_accuracy = np.full(n_b, np.nan, dtype=np.float64)
        random_throughput_ratio = np.full(n_b, np.nan, dtype=np.float64)
        random_BME = np.full(n_b, np.nan, dtype=np.float64)

    print("\nB | overhead | MIMOGS_BME | baseline_BME | random_BME | MIMOGS_align | MIMOGS_thr")
    for i, B in enumerate(B_values):
        rb = random_BME[i]
        rb_str = f"{rb:.6f}" if np.isfinite(rb) else "N/A"
        print(
            f"{int(B):2d} | {overhead[i]:.4f} | {mimogs_BME[i]:.6f} | "
            f"{baseline_BME[i]:.6f} | {rb_str:>8} | {mimogs_alignment_accuracy[i]:.6f} | {mimogs_throughput_ratio[i]:.6f}"
        )

    mat_dict = {
        "B_values": B_values.astype(np.int32),
        "overhead": overhead.astype(np.float64),
        "overhead_denominator": np.array([int(args.overhead_denominator)], dtype=np.int32),
        "mimogs_alignment_accuracy": mimogs_alignment_accuracy.astype(np.float64),
        "mimogs_throughput_ratio": mimogs_throughput_ratio.astype(np.float64),
        "mimogs_BME": mimogs_BME.astype(np.float64),
        "baseline_alignment_accuracy": baseline_alignment_accuracy.astype(np.float64),
        "baseline_throughput_ratio": baseline_throughput_ratio.astype(np.float64),
        "baseline_BME": baseline_BME.astype(np.float64),
        "random_alignment_accuracy": random_alignment_accuracy.astype(np.float64),
        "random_throughput_ratio": random_throughput_ratio.astype(np.float64),
        "random_BME": random_BME.astype(np.float64),
        "random_trials": np.array([int(args.random_trials)], dtype=np.int32),
        "num_test_used": np.array([int(num_test_used)], dtype=np.int32),
        "total_beam_pairs": np.array([int(total_beam_pairs)], dtype=np.int32),
        "beam_rows": np.array([int(beam_rows)], dtype=np.int32),
        "beam_cols": np.array([int(beam_cols)], dtype=np.int32),
        "model_path": np.array([args.output_dir], dtype=object),
        "source_path": np.array([args.source_path], dtype=object),
        "checkpoint_path": np.array([args.checkpoint_path], dtype=object),
        "metric_description": np.array(
            ["BME(B)=alignment_accuracy(B)*throughput_ratio(B)*(1-B/64), throughput_ratio=|H_chosen|^2/|H_best|^2"],
            dtype=object,
        ),
        "method_descriptions": np.array(
            [
                "MIMOGS: top-B beams ranked by predicted rendered magnitude, then best measured beam among probed candidates.",
                "Baseline: top-B beams ranked by training-set average beam power prior.",
                "Random: B uniformly random beams, averaged over trials.",
            ],
            dtype=object,
        ),
    }

    savemat(args.mat_path, mat_dict)
    print(f"\n[Done] Saved MAT file: {args.mat_path}")


def ensure_comparison_m_exists():
    path = os.path.abspath("comparison.m")
    if not os.path.exists(path):
        print(f"[Warning] comparison.m not found at repo root: {path}")


def main():
    parser, model_defaults = build_parser()
    args = merge_cfg_and_cli(parser)
    args = apply_required_defaults(args, model_defaults)
    ensure_comparison_m_exists()
    evaluate(args)


if __name__ == "__main__":
    main()
