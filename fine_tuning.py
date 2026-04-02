import os
# import csv
import copy
import random
from argparse import ArgumentParser
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from gaussian_renderer import render
from scene import Scene, GaussianModel
from train import (
    prepare_output_dir,
    save_run_args_txt,
    evaluate_and_save_random_test_samples,
    get_avg_opacity,
)
from utils.general_utils import safe_state
from utils.loss import hybrid_magnitude_loss, magnitude_mse_loss, normalize_mag_map


def make_timestamp_model_path(base_dir: str = "outputs", prefix: str = "fine_tune") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{prefix}_{timestamp}")


def namespace_from_dict(d: Dict) -> SimpleNamespace:
    return SimpleNamespace(**dict(d))


def clone_gaussian_state(gaussian_state):
    cloned = []
    for x in gaussian_state:
        if torch.is_tensor(x):
            cloned.append(x.detach().cpu().clone())
        else:
            cloned.append(copy.deepcopy(x))
    return tuple(cloned)


def strip_optimizer_state(gaussian_state):
    state = list(gaussian_state)
    # capture() layout in GaussianModel:
    # 0 target_gaussians
    # 1 optimizer_type
    # 2 init_range
    # 3 xyz
    # 4 scaling
    # 5 rotation
    # 6 opacity
    # 7 gain_mag
    # 8 xyz_gradient_accum
    # 9 grad_denom
    # 10 importance_accum
    # 11 importance_denom
    # 12 optimizer_state_dict
    # 13 dynamic_gain_net_state_dict
    # 14 dynamic_gain_optimizer_state_dict
    state[12] = None
    state[14] = None
    return tuple(state)


def parse_mix_ratio(s: str) -> Tuple[int, int]:
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"mix_ratio must look like '1:1' or '2:1', got {s}")
    hard_n = int(parts[0])
    normal_n = int(parts[1])
    if hard_n <= 0 or normal_n <= 0:
        raise ValueError(f"mix_ratio entries must be positive, got {s}")
    return hard_n, normal_n


def build_mixed_epoch_indices(
    hard_indices: Sequence[int],
    normal_indices: Sequence[int],
    epoch_size: int,
    mix_ratio: str,
    rng: random.Random,
) -> List[int]:
    hard_n, normal_n = parse_mix_ratio(mix_ratio)
    total_units = hard_n + normal_n
    target_hard = int(round(epoch_size * hard_n / total_units))
    target_hard = max(hard_n, min(epoch_size, target_hard))
    target_normal = max(0, epoch_size - target_hard)

    if len(hard_indices) == 0:
        raise ValueError("No hard indices were selected.")
    if len(normal_indices) == 0:
        hard_draw = rng.choices(list(hard_indices), k=epoch_size)
        rng.shuffle(hard_draw)
        return hard_draw

    hard_draw = rng.choices(list(hard_indices), k=target_hard)
    if target_normal <= len(normal_indices):
        normal_draw = rng.sample(list(normal_indices), k=target_normal)
    else:
        normal_draw = rng.choices(list(normal_indices), k=target_normal)

    mixed = hard_draw + normal_draw
    rng.shuffle(mixed)
    return mixed


def compute_rows_for_indices(
    dataset,
    indices: Sequence[int],
    gaussians: GaussianModel,
    tx_pos: torch.Tensor,
    beam_rows: int,
    beam_cols: int,
) -> Tuple[List[Dict[str, float]], float, float, float]:
    rows: List[Dict[str, float]] = []
    device = gaussians.get_xyz.device

    with torch.no_grad():
        for idx in indices:
            magnitude, rx_pos = dataset[idx]
            magnitude = magnitude.to(device).reshape(beam_rows, beam_cols)
            rx_pos = rx_pos.to(device)

            out = render(
                rx_pos=rx_pos,
                tx_pos=tx_pos,
                pc=gaussians,
                rx_shape=(2, 2),
                tx_shape=(4, 4),
                normalize_beam_weights=False,
                weight_floor=0.0,
            )
            pred_mag = out["render"]

            gt_shape = normalize_mag_map(magnitude)
            pred_shape = normalize_mag_map(pred_mag)

            loss_val = magnitude_mse_loss(pred_shape, gt_shape).item()
            zero_val = torch.mean(gt_shape ** 2).item()
            ratio_val = loss_val / max(zero_val, 1e-12)

            rows.append(
                {
                    "idx": int(idx),
                    "loss": float(loss_val),
                    "zero": float(zero_val),
                    "ratio_to_zero": float(ratio_val),
                }
            )

    mean_loss = sum(r["loss"] for r in rows) / len(rows)
    mean_zero = sum(r["zero"] for r in rows) / len(rows)
    mean_ratio = sum(r["ratio_to_zero"] for r in rows) / len(rows)
    return rows, mean_loss, mean_zero, mean_ratio


# def write_hard_mining_csv(csv_path: str, rows: Sequence[Dict[str, float]]):
#     with open(csv_path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["rank", "idx", "loss", "zero_loss", "ratio_to_zero"])
#         for rank, row in enumerate(rows):
#             writer.writerow(
#                 [
#                     rank,
#                     row["idx"],
#                     f"{row['loss']:.8f}",
#                     f"{row['zero']:.8f}",
#                     f"{row['ratio_to_zero']:.8f}",
#                 ]
#             )


# def write_debug_csv(csv_path: str, init_rows: Sequence[Dict[str, float]], final_rows: Sequence[Dict[str, float]]):
#     with open(csv_path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(
#             [
#                 "idx",
#                 "init_loss",
#                 "final_loss",
#                 "zero_loss",
#                 "init_ratio_to_zero",
#                 "final_ratio_to_zero",
#             ]
#         )
#         for r0, r1 in zip(init_rows, final_rows):
#             writer.writerow(
#                 [
#                     r0["idx"],
#                     f"{r0['loss']:.8f}",
#                     f"{r1['loss']:.8f}",
#                     f"{r1['zero']:.8f}",
#                     f"{r0['ratio_to_zero']:.8f}",
#                     f"{r1['ratio_to_zero']:.8f}",
#                 ]
#             )


def apply_finetune_lr_scaling(opt_params, lr_scale: float):
    for field in [
        "position_lr_init",
        "position_lr_final",
        "opacity_lr",
        "opacity_lr_final",
        "scaling_lr",
        "rotation_lr",
        "gain_lr",
        "gain_lr_final",
        "dynamic_gain_lr",
        "dynamic_gain_lr_final",
    ]:
        if hasattr(opt_params, field):
            setattr(opt_params, field, getattr(opt_params, field) * lr_scale)


def set_output_path(model_params, explicit_output_path: str = ""):
    if explicit_output_path:
        model_params.model_path = os.path.abspath(explicit_output_path)
    else:
        model_params.model_path = os.path.abspath(make_timestamp_model_path("outputs", "fine_tune"))


def training(args):
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model_params = namespace_from_dict(checkpoint["model_params"])
    opt_params = namespace_from_dict(checkpoint["opt_params"])

    if args.source_path:
        model_params.source_path = os.path.abspath(args.source_path)
    if args.data_device:
        model_params.data_device = args.data_device
    set_output_path(model_params, args.output_path)

    prepare_output_dir(model_params.model_path)
    save_run_args_txt(model_params.model_path, model_params, opt_params, args)

    device = torch.device(model_params.data_device if torch.cuda.is_available() else "cpu")

    base_state = checkpoint["gaussians"]
    if not args.reuse_optimizer_state:
        base_state = strip_optimizer_state(base_state)

    gaussians = GaussianModel(
        target_gaussians=10_000,
        optimizer_type=getattr(opt_params, "optimizer_type", "default"),
        device=str(device),
        init_range=getattr(opt_params, "init_range", 1),
    )
    scene = Scene(model_params, gaussians)

    tx_pos = torch.tensor(scene.bs_position, dtype=torch.float32, device=device)

    # Fixed train-subset diagnostics to match train.py
    debug_indices = list(range(256))

    # Stage-2 schedule settings
    ft_epochs = args.ft_epochs
    epoch_size = len(scene.train_set)
    total_iterations = epoch_size * ft_epochs
    apply_finetune_lr_scaling(opt_params, args.lr_scale)
    opt_params.iterations = total_iterations
    opt_params.position_lr_max_steps = max(1, int(args.position_lr_frac * total_iterations))

    gaussians.restore(base_state, opt_params)

    # Optional statistics reset; densify/prune is off in this script
    if hasattr(gaussians, "_reset_statistics"):
        gaussians._reset_statistics()

    print(f"[FineTune] Device: {device}")
    print(f"[FineTune] Source path: {getattr(model_params, 'source_path', '')}")
    print(f"[FineTune] Checkpoint path: {os.path.abspath(args.checkpoint_path)}")
    print(f"[FineTune] Output path: {model_params.model_path}")
    print(f"[FineTune] Train set size: {len(scene.train_set)}")
    print(f"[FineTune] Test set size: {len(scene.test_set)}")
    print(f"[FineTune] Fine-tune epochs: {ft_epochs}")
    print(f"[FineTune] Total iterations: {total_iterations}")
    print(f"[FineTune] LR scale: {args.lr_scale}")
    print(f"[FineTune] position_lr_max_steps: {opt_params.position_lr_max_steps}")

    # --------------------------------------------------
    # Initial debug stats on the fixed train subset
    # --------------------------------------------------
    init_rows, init_loss, zero_loss, init_ratio = compute_rows_for_indices(
        dataset=scene.train_set,
        indices=debug_indices,
        gaussians=gaussians,
        tx_pos=tx_pos,
        beam_rows=scene.beam_rows,
        beam_cols=scene.beam_cols,
    )
    print(f"[Debug] subset mean init loss: {init_loss:.8f}")
    print(f"[Debug] subset mean zero baseline: {zero_loss:.8f}")
    print(f"[Debug] subset mean init ratio_to_zero: {init_ratio:.8f}")

    # --------------------------------------------------
    # Hard example mining on TRAIN SET ONLY
    # --------------------------------------------------
    all_train_indices = list(range(len(scene.train_set)))
    print(f"[HardMining] Scoring all {len(all_train_indices)} train samples...")
    train_rows, _, _, _ = compute_rows_for_indices(
        dataset=scene.train_set,
        indices=all_train_indices,
        gaussians=gaussians,
        tx_pos=tx_pos,
        beam_rows=scene.beam_rows,
        beam_cols=scene.beam_cols,
    )
    train_rows_sorted = sorted(train_rows, key=lambda r: r["ratio_to_zero"], reverse=True)

    requested_hard = int(round(args.hard_fraction * len(train_rows_sorted)))
    requested_hard = max(args.min_hard, requested_hard)
    if args.max_hard > 0:
        requested_hard = min(requested_hard, args.max_hard)
    requested_hard = min(requested_hard, len(train_rows_sorted) - 1)

    hard_indices = [r["idx"] for r in train_rows_sorted[:requested_hard]]
    hard_set = set(hard_indices)
    normal_indices = [idx for idx in all_train_indices if idx not in hard_set]

    print(f"[HardMining] Selected hard samples: {len(hard_indices)}")
    print(f"[HardMining] Selected normal pool: {len(normal_indices)}")
    print(
        f"[HardMining] Hard ratio_to_zero min/median/max: "
        f"{train_rows_sorted[requested_hard-1]['ratio_to_zero']:.8f} / "
        f"{train_rows_sorted[requested_hard//2]['ratio_to_zero']:.8f} / "
        f"{train_rows_sorted[0]['ratio_to_zero']:.8f}"
    )

    # write_hard_mining_csv(
    #     os.path.join(model_params.model_path, "train_hard_mining.csv"),
    #     train_rows_sorted,
    # )

    # --------------------------------------------------
    # Stage-2 fine-tuning
    # --------------------------------------------------
    rng = random.Random(args.seed)
    best_debug_loss = init_loss
    best_epoch = 0
    best_state = clone_gaussian_state(gaussians.capture())

    iteration = 0
    ema_loss = 0.0
    progress_bar = tqdm(total=total_iterations, desc="Fine-tuning progress")

    for epoch in range(ft_epochs):
        epoch_indices = build_mixed_epoch_indices(
            hard_indices=hard_indices,
            normal_indices=normal_indices,
            epoch_size=epoch_size,
            mix_ratio=args.mix_ratio,
            rng=rng,
        )

        for idx in epoch_indices:
            iteration += 1
            gaussians.update_learning_rate(iteration)

            magnitude, rx_pos = scene.train_set[idx]
            magnitude = magnitude.to(device)
            rx_pos = rx_pos.to(device)
            gt_mag = magnitude.reshape(scene.beam_rows, scene.beam_cols)

            out = render(
                rx_pos=rx_pos,
                tx_pos=tx_pos,
                pc=gaussians,
                rx_shape=(2, 2),
                tx_shape=(4, 4),
                normalize_beam_weights=False,
                weight_floor=0.0,
            )
            pred_mag = out["render"]
            importance = out["per_gaussian_importance"]

            loss, abs_loss_dbg, topk_loss_dbg = hybrid_magnitude_loss(
                pred_mag,
                gt_mag,
                topk_ratio=0.125,
                eps=1e-8,
                return_terms=True,
            )

            gaussians.optimizer.zero_grad(set_to_none=True)
            gaussians.dynamic_gain_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gaussians.accumulate_training_stats(importance=importance)
            gaussians.optimizer.step()
            gaussians.dynamic_gain_optimizer.step()

            dyn_grad_norm = 0.0
            for p in gaussians.dynamic_gain_net.parameters():
                if p.grad is not None:
                    dyn_grad_norm += p.grad.norm().item()

            if iteration > 1000 and iteration % 1000 == 0:
                xyz_grad = gaussians._xyz.grad.norm().item() if gaussians._xyz.grad is not None else 0.0
                opacity_grad = gaussians._opacity.grad.norm().item() if gaussians._opacity.grad is not None else 0.0
                scaling_grad = gaussians._scaling.grad.norm().item() if gaussians._scaling.grad is not None else 0.0
                rotation_grad = gaussians._rotation.grad.norm().item() if gaussians._rotation.grad is not None else 0.0
                print(
                    f"grad xyz={xyz_grad:.3e}, "
                    f"opacity={opacity_grad:.3e}, "
                    f"scaling={scaling_grad:.3e}, "
                    f"rotation={rotation_grad:.3e}, "
                    f"dyn_gain={dyn_grad_norm:.3e}"
                )

            if iteration > 0 and iteration % 1000 == 0:
                avg_opacity = get_avg_opacity(gaussians)
                print(
                    f"nums of gaussians: {gaussians.get_xyz.shape[0]}, "
                    f"Avg opacity: {avg_opacity:.4f}, "
                    f"abs_loss: {float(abs_loss_dbg):.8f}, "
                    f"topk_loss: {float(topk_loss_dbg):.8f}, "
                )

            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss:.8f}", "Epoch": epoch + 1})
                progress_bar.update(10)

        # end epoch: train-subset debug only
        epoch_rows, epoch_loss, _, epoch_ratio = compute_rows_for_indices(
            dataset=scene.train_set,
            indices=debug_indices,
            gaussians=gaussians,
            tx_pos=tx_pos,
            beam_rows=scene.beam_rows,
            beam_cols=scene.beam_cols,
        )
        print(
            f"[Epoch {epoch+1:03d}] debug_subset_loss={epoch_loss:.8f}, "
            f"debug_subset_ratio_to_zero={epoch_ratio:.8f}"
        )
        if epoch_loss < best_debug_loss:
            best_debug_loss = epoch_loss
            best_epoch = epoch + 1
            best_state = clone_gaussian_state(gaussians.capture())
            print(f"[Best] Updated best debug loss at epoch {best_epoch}: {best_debug_loss:.8f}")

    progress_bar.close()

    # Restore best train-subset checkpoint before final reporting / saving
    gaussians.restore(best_state, opt_params)

    final_rows, final_loss, final_zero, final_ratio = compute_rows_for_indices(
        dataset=scene.train_set,
        indices=debug_indices,
        gaussians=gaussians,
        tx_pos=tx_pos,
        beam_rows=scene.beam_rows,
        beam_cols=scene.beam_cols,
    )

    print(f"[Best] best_epoch_by_debug_loss: {best_epoch}")
    print(f"[Debug] subset mean final loss: {final_loss:.8f}")
    print(f"[Debug] loss ratio final/init: {final_loss / max(init_loss, 1e-12):.8f}")
    print(f"[Debug] loss ratio final/zero: {final_loss / max(zero_loss, 1e-12):.8f}")
    print(f"[Debug] subset mean final ratio_to_zero: {final_ratio:.8f}")

    final_ratios = sorted(r["ratio_to_zero"] for r in final_rows)
    print(f"[Debug] per-sample final ratio_to_zero min: {final_ratios[0]:.8f}")
    print(f"[Debug] per-sample final ratio_to_zero median: {final_ratios[len(final_ratios)//2]:.8f}")
    print(f"[Debug] per-sample final ratio_to_zero max: {final_ratios[-1]:.8f}")

    # debug_csv_path = os.path.join(model_params.model_path, "debug_subset_losses.csv")
    # write_debug_csv(debug_csv_path, init_rows, final_rows)
    # print(f"[Debug] saved per-sample debug csv to: {debug_csv_path}")

    point_cloud_path = os.path.join(model_params.model_path, "point_cloud", "point_cloud.ply")
    gaussians.save_ply(point_cloud_path)
    print(f"[Save] Saved point cloud to {point_cloud_path}")

    model_ckpt = os.path.join(model_params.model_path, "model.pth")
    torch.save(
        {
            "iteration": iteration,
            "gaussians": gaussians.capture(),
            "model_params": vars(model_params),
            "opt_params": vars(opt_params),
            "fine_tune_args": vars(args),
            "best_epoch_by_debug_loss": best_epoch,
            "hard_indices": hard_indices,
        },
        model_ckpt,
    )
    print(f"[Save] Saved model checkpoint to {model_ckpt}")

    # TEST SET is used only here for evaluation / visualization
    evaluate_and_save_random_test_samples(
        scene=scene,
        gaussians=gaussians,
        model_params=model_params,
        num_samples=args.eval_num_samples,
    )
    print("[FineTune] Done.")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train-only hard-example fine-tuning for MIMOGS")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="outputs/20260401_141614/model.pth",
        help="Starting checkpoint (best stage-1 model).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Optional explicit output directory. Default: outputs/fine_tune_YYYYMMDD_HHMMSS",
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="",
        help="Optional dataset override. If empty, use the source_path stored in the checkpoint.",
    )
    parser.add_argument(
        "--data_device",
        type=str,
        default="",
        help="Optional device override, e.g. cuda or cpu.",
    )
    parser.add_argument("--ft_epochs", type=int, default=20)
    parser.add_argument("--hard_fraction", type=float, default=0.1)
    parser.add_argument("--min_hard", type=int, default=256)
    parser.add_argument("--max_hard", type=int, default=2048)
    parser.add_argument(
        "--mix_ratio",
        type=str,
        default="1:1",
        help="Hard:normal sampling ratio for stage-2 fine-tuning, e.g. 1:1 or 2:1.",
    )
    parser.add_argument(
        "--lr_scale",
        type=float,
        default=0.5,
        help="Global multiplicative LR scale applied to checkpoint opt_params for stage-2 fine-tuning.",
    )
    parser.add_argument(
        "--position_lr_frac",
        type=float,
        default=0.6,
        help="position_lr_max_steps = int(position_lr_frac * total_iterations)",
    )
    parser.add_argument(
        "--reuse_optimizer_state",
        action="store_true",
        default=False,
        help="If set, reuse optimizer states stored in the checkpoint. By default, stage-2 starts with fresh optimizers.",
    )
    parser.add_argument("--eval_num_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    safe_state(args.quiet)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    training(args)
