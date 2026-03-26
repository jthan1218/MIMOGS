import os
import random
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from torch.utils.data import DataLoader, Subset


def magnitude_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def normalized_magnitude_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    abs_loss = magnitude_mse_loss(pred, target)
    target_power = torch.mean(target ** 2)
    return abs_loss / (target_power + eps)


def log_magnitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mag_scale: float = 1e-3,
) -> torch.Tensor:
    pred_log = torch.log1p(pred / mag_scale)
    target_log = torch.log1p(target / mag_scale)
    return torch.mean((pred_log - target_log) ** 2)


# def hybrid_magnitude_loss(
#     pred: torch.Tensor,
#     target: torch.Tensor,
#     alpha: float = 0.2,
#     beta: float = 0.3,
#     gamma: float = 0.5,
#     eps: float = 1e-4,
#     mag_scale: float = 1e-3,
#     return_terms: bool = False,
# ):
#     abs_loss = magnitude_mse_loss(pred, target)
#     rel_loss = normalized_magnitude_mse_loss(pred, target, eps=eps)
#     log_loss = log_magnitude_loss(pred, target, mag_scale=mag_scale)

#     total_loss = alpha * abs_loss + beta * rel_loss + gamma * log_loss

#     if return_terms:
#         target_power = torch.mean(target ** 2)
#         return total_loss, abs_loss.detach(), rel_loss.detach(), log_loss.detach(), target_power.detach()

#     return total_loss

# def hybrid_magnitude_loss(
#     pred: torch.Tensor,
#     target: torch.Tensor,
#     alpha: float = 0.3,
#     beta: float = 0.2,
#     gamma: float = 0.5,
#     eps: float = 1e-8,
#     mag_scale: float = 0.05,
#     return_terms: bool = False,
# ):
#     """
#     Version 0: sample-wise max-normalized magnitude shape loss.
#     Both pred and target are normalized by their own sample max.
#     """
#     pred_n = normalize_mag_map(pred, eps=eps)
#     target_n = normalize_mag_map(target, eps=eps)

#     abs_loss = magnitude_mse_loss(pred_n, target_n)
#     rel_loss = normalized_magnitude_mse_loss(pred_n, target_n, eps=eps)
#     log_loss = log_magnitude_loss(pred_n, target_n, mag_scale=mag_scale)

#     total_loss = alpha * abs_loss + beta * rel_loss + gamma * log_loss

#     if return_terms:
#         target_power = torch.mean(target_n ** 2)
#         return total_loss, abs_loss.detach(), rel_loss.detach(), log_loss.detach(), target_power.detach()

#     return total_loss

def hybrid_magnitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.4,
    beta: float = 0.05,
    gamma: float = 0.2,
    lambda_topk: float = 0.10,
    lambda_bg: float = 0.01,
    lambda_rank: float = 0.02,
    topk_ratio: float = 0.125,
    bg_threshold: float = 0.05,
    rank_margin: float = 0.05,
    num_neg: int = 16,
    eps: float = 1e-8,
    mag_scale: float = 0.05,
    return_terms: bool = False,
):
    # pred는 raw scale 유지
    pred_n = pred

    # target만 sample-wise normalization
    target_n = normalize_mag_map(target, eps=eps)

    abs_loss = magnitude_mse_loss(pred_n, target_n)
    rel_loss = normalized_magnitude_mse_loss(pred_n, target_n, eps=eps)
    log_loss = log_magnitude_loss(pred_n, target_n, mag_scale=mag_scale)

    topk_loss = topk_shape_loss(pred_n, target_n, topk_ratio=topk_ratio)
    bg_loss = background_suppression_loss(pred_n, target_n, bg_threshold=bg_threshold)
    rank_loss = ranking_separation_loss(
        pred_n, target_n, topk_ratio=topk_ratio, num_neg=num_neg, margin=rank_margin
    )

    # total_loss = (
    # 0.7 * abs_loss
    # + 0.3 * topk_loss
    # )   

    # total_loss = (
    #     alpha * abs_loss
    #     + beta * rel_loss
    #     + gamma * log_loss
    #     + lambda_topk * topk_loss
    #     + lambda_bg * bg_loss
    #     + lambda_rank * rank_loss
    # )

    total_loss = (
    0.45 * abs_loss
    + 0.20 * topk_loss
    + 0.20 * bg_loss
    + 0.15 * log_loss
    )

    if return_terms:
        target_power = torch.mean(target_n ** 2)
        return (
            total_loss,
            abs_loss.detach(),
            rel_loss.detach(),
            log_loss.detach(),
            topk_loss.detach(),
            bg_loss.detach(),
            rank_loss.detach(),
            target_power.detach(),
        )

    return total_loss

def normalize_mag_map(x: torch.Tensor, eps:float = 1e-8) -> torch.Tensor:
    return x / (torch.amax(x) + eps)

def topk_shape_loss(
    pred_n: torch.Tensor,
    target_n: torch.Tensor,
    topk_ratio: float = 0.125,
) -> torch.Tensor:
    pred_flat = pred_n.reshape(-1)
    target_flat = target_n.reshape(-1)

    k = max(1, int(round(topk_ratio * target_flat.numel())))
    topk_idx = torch.topk(target_flat, k=k, largest=True).indices

    return torch.mean((pred_flat[topk_idx] - target_flat[topk_idx]) ** 2)

def background_suppression_loss(
    pred_n: torch.Tensor,
    target_n: torch.Tensor,
    bg_threshold: float = 0.05,
) -> torch.Tensor:
    """
    Penalize prediction leakage on weak/background beams.
    target_n is already sample-wise normalized to [0,1].
    """
    bg_mask = (target_n <= bg_threshold).float()
    denom = bg_mask.sum().clamp_min(1.0)
    return ((pred_n ** 2) * bg_mask).sum() / denom


def ranking_separation_loss(
    pred_n: torch.Tensor,
    target_n: torch.Tensor,
    topk_ratio: float = 0.125,
    num_neg: int = 16,
    margin: float = 0.05,
) -> torch.Tensor:
    """
    Hard-negative ranking loss.
    Positive  : GT top-k beams
    Negative  : non-topk beams with the largest predicted values
    """
    pred_flat = pred_n.reshape(-1)
    target_flat = target_n.reshape(-1)

    total_num = target_flat.numel()
    k = max(1, int(round(topk_ratio * total_num)))

    # 1) Positive set = GT top-k
    pos_idx = torch.topk(target_flat, k=k, largest=True).indices

    # 2) Negative candidate pool = everything except GT top-k
    neg_mask = torch.ones(total_num, dtype=torch.bool, device=target_flat.device)
    neg_mask[pos_idx] = False
    neg_pool_idx = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)

    if neg_pool_idx.numel() == 0:
        return pred_flat.new_zeros(())

    # 3) Hard negatives = among non-topk beams, choose the ones
    #    that the model predicted too strongly
    num_neg = min(num_neg, neg_pool_idx.numel())
    hard_order = torch.topk(pred_flat[neg_pool_idx], k=num_neg, largest=True).indices
    neg_idx = neg_pool_idx[hard_order]

    # 4) Margin ranking
    pos = pred_flat[pos_idx].unsqueeze(1)   # (k, 1)
    neg = pred_flat[neg_idx].unsqueeze(0)   # (1, num_neg)

    return torch.relu(margin - (pos - neg)).mean()

# def collect_hard_example_indices(
#     scene,
#     gaussians,
#     tx_pos,
#     device,
#     fraction: float = 0.2,
#     min_count: int = 512,
#     max_count: int = 2048,
# ):
#     """
#     Evaluate the whole TRAIN set once, rank samples by ratio_to_zero,
#     and return the hardest indices.
#     """
#     rows = []

#     with torch.no_grad():
#         for idx in range(len(scene.train_set)):
#             magnitude, rx_pos = scene.train_set[idx]

#             gt_mag = magnitude.to(device).reshape(scene.beam_rows, scene.beam_cols)
#             rx_pos = rx_pos.to(device)

#             out = render(
#                 rx_pos=rx_pos,
#                 tx_pos=tx_pos,
#                 pc=gaussians,
#                 rx_shape=(2, 2),
#                 tx_shape=(4, 4),
#                 normalize_beam_weights=False,
#                 weight_floor=0.0,
#             )
#             pred_mag = out["render"]

#             loss_val = magnitude_mse_loss(pred_mag, gt_mag).item()
#             zero_val = torch.mean(gt_mag ** 2).item()
#             ratio_val = loss_val / max(zero_val, 1e-12)

#             rows.append({
#                 "idx": idx,
#                 "loss": loss_val,
#                 "zero": zero_val,
#                 "ratio_to_zero": ratio_val,
#             })

#     rows.sort(key=lambda r: r["ratio_to_zero"], reverse=True)

#     k = int(len(rows) * fraction)
#     k = max(k, min_count)
#     k = min(k, max_count, len(rows))

#     hard_indices = [r["idx"] for r in rows[:k]]
#     return hard_indices, rows


# def run_hard_example_finetune(
#     scene,
#     gaussians,
#     tx_pos,
#     device,
#     model_params,
#     iteration: int,
#     hard_indices,
#     hard_epochs: int = 5,
#     lr_mult: float = 0.3,
# ):
#     """
#     Fine-tune only on hardest TRAIN samples for a few epochs.
#     Uses the same loss as the main training loop.
#     """
#     if len(hard_indices) == 0:
#         return iteration

#     hard_set = Subset(scene.train_set, hard_indices)
#     hard_loader = DataLoader(
#         hard_set,
#         batch_size=1,
#         shuffle=True,
#         num_workers=0,
#     )

#     # save current learning rates
#     old_lrs = [pg["lr"] for pg in gaussians.optimizer.param_groups]
#     for pg in gaussians.optimizer.param_groups:
#         pg["lr"] = pg["lr"] * lr_mult

#     hard_total_iters = len(hard_loader) * hard_epochs
#     progress_bar = tqdm(total=hard_total_iters, desc="Hard-example fine-tune")
#     ema_loss = 0.0

#     for _ in range(hard_epochs):
#         for batch in hard_loader:
#             iteration += 1
#             gaussians.update_learning_rate(iteration)

#             magnitude, rx_pos = batch

#             magnitude = magnitude.squeeze(0).to(device)
#             rx_pos = rx_pos.squeeze(0).to(device)

#             gt_mag = magnitude.reshape(scene.beam_rows, scene.beam_cols)

#             out = render(
#                 rx_pos=rx_pos,
#                 tx_pos=tx_pos,
#                 pc=gaussians,
#                 rx_shape=(2, 2),
#                 tx_shape=(4, 4),
#                 normalize_beam_weights=False,
#                 weight_floor=0.0,
#             )

#             pred_mag = out["render"]
#             importance = out["per_gaussian_importance"]

#             loss, abs_loss_dbg, rel_loss_dbg, log_loss_dbg, gt_power_dbg = hybrid_magnitude_loss(
#                 pred_mag,
#                 gt_mag,
#                 alpha=0.2,
#                 beta=0.3,
#                 gamma=0.5,
#                 eps=1e-4,
#                 mag_scale=1e-3,
#                 return_terms=True,
#             )

#             gaussians.optimizer.zero_grad(set_to_none=True)
#             loss.backward()
#             gaussians.accumulate_training_stats(importance=importance)
#             gaussians.optimizer.step()

#             ema_loss = 0.4 * float(loss.item()) + 0.6 * ema_loss
#             progress_bar.set_postfix({"Loss": f"{ema_loss:.8f}"})
#             progress_bar.update(1)

#     progress_bar.close()

#     # restore original learning rates
#     for pg, old_lr in zip(gaussians.optimizer.param_groups, old_lrs):
#         pg["lr"] = old_lr

#     return iteration


def prepare_output_dir(model_path: str):
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(os.path.join(model_path, "point_cloud"), exist_ok=True)
    os.makedirs(os.path.join(model_path, "pred_compare"), exist_ok=True)


def save_run_args_txt(model_path: str, model_params, opt_params, raw_args):
    txt_path = os.path.join(model_path, "run_args.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("[Model Params]\n")
        for k, v in sorted(vars(model_params).items()):
            f.write(f"{k}: {v}\n")

        f.write("\n[Optimization Params]\n")
        for k, v in sorted(vars(opt_params).items()):
            f.write(f"{k}: {v}\n")

        f.write("\n[RawArgs Namespace]\n")
        for k, v in sorted(vars(raw_args).items()):
            f.write(f"{k}: {v}\n")

def make_timestamp_model_path(base_dir: str = "outputs") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, timestamp)


def evaluate_and_save_random_test_samples(
    scene: Scene,
    gaussians: GaussianModel,
    model_params,
    num_samples: int = 50,
):
    save_dir = os.path.join(model_params.model_path, "pred_compare")
    os.makedirs(save_dir, exist_ok=True)

    total = len(scene.test_set)
    num_samples = min(num_samples, total)
    rng = random.Random(12345)
    indices = rng.sample(range(total), num_samples)

    tx_pos = torch.tensor(
        scene.bs_position,
        dtype=torch.float32,
        device=gaussians.get_xyz.device,
    )

    print(f"[Evaluation] Rendering {num_samples} random test samples...")

    with torch.no_grad():
        for rank, idx in enumerate(indices):
            magnitude, rx_pos = scene.test_set[idx]

            rx_pos = rx_pos.to(gaussians.get_xyz.device)
            magnitude = magnitude.to(gaussians.get_xyz.device)
            magnitude = magnitude.reshape(scene.beam_rows, scene.beam_cols)

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

            # gt_mag_np = magnitude.detach().cpu().numpy()
            # pred_mag_np = pred_mag.detach().cpu().numpy()

            gt_mag_np = normalize_mag_map(magnitude).detach().cpu().numpy()
            pred_mag_np = pred_mag.detach().cpu().numpy()

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            im0 = axes[0].imshow(gt_mag_np, aspect="equal", interpolation="nearest")
            axes[0].set_title("Ground Truth Shape (sample-wise normalized)")
            plt.colorbar(im0, ax=axes[0], fraction=0.03, pad=0.04)

            im1 = axes[1].imshow(pred_mag_np, aspect="equal", interpolation="nearest")
            axes[1].set_title("Predicted Shape (raw scale)")
            plt.colorbar(im1, ax=axes[1], fraction=0.03, pad=0.04)

            for ax in axes.ravel():
                ax.set_xlabel("Tx beam index")
                ax.set_ylabel("Rx beam index")
                ax.set_aspect("equal")

            fig.suptitle(f"Test sample idx={idx}", fontsize=12)
            fig.tight_layout()

            fig_path = os.path.join(save_dir, f"{rank:02d}.png")
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)

    print(f"[Eval] Saved comparison figures to {save_dir}")

def get_avg_opacity(gaussians) -> float:
    with torch.no_grad():
        if hasattr(gaussians, "get_opacity"):
            opacity = gaussians.get_opacity
        elif hasattr(gaussians, "_opacity"):
            opacity = torch.sigmoid(gaussians._opacity)
        elif hasattr(gaussians, "opacity"):
            opacity = gaussians.opacity
        else:
            return float("nan")

        if torch.is_complex(opacity):
            opacity = torch.abs(opacity)

        return float(opacity.detach().mean().item())

def _finite_ratio(x: torch.Tensor) -> float:
    if torch.is_complex(x):
        xr = torch.view_as_real(x.detach())
    else:
        xr = x.detach()
    return float(torch.isfinite(xr).float().mean().item())

def assert_finite(name: str, x: torch.Tensor, iteration: int):
    xr = torch.view_as_real(x) if torch.is_complex(x) else x
    if not torch.isfinite(xr).all():
        raise RuntimeError(
            f"[NaN/Inf detected] {name} at iter={iteration}, "
            f"finite_ratio={_finite_ratio(x):.6f}, "
            f"shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}"
        )



########################################################
# Training loop
########################################################
def training(model_params, opt_params, raw_args):
    device = torch.device(model_params.data_device if torch.cuda.is_available() else "cpu")

    if not getattr(model_params, "model_path", None):
        model_params.model_path = make_timestamp_model_path("outputs")
    
    prepare_output_dir(model_params.model_path)
    save_run_args_txt(model_params.model_path, model_params, opt_params, raw_args)

    gaussians = GaussianModel(
        target_gaussians = 5_000,
        optimizer_type = opt_params.optimizer_type,
        device = str(device),
        init_range = 1,
    )

    scene = Scene(model_params, gaussians)

    # --------------------------------------------------
    # Debug: overfit fixed 16 train samples
    # --------------------------------------------------
    fixed_subset_debug = False
    fixed_indices = list(range(256))

    if fixed_subset_debug:
        subset = Subset(scene.train_set, fixed_indices)
        scene.train_iter = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        scene.num_epochs = 1000
    ########################################################


    if getattr(model_params, "init_mode", "random") == "vertices" and getattr(model_params, "vertices_path",""):
        gaussians.gaussian_init(vertices_path=model_params.vertices_path)
    else:
        gaussians.gaussian_init(vertices_path=None)

    gaussians.training_setup(opt_params)

    tx_pos = torch.tensor(
        scene.bs_position,
        dtype=torch.float32,
        device = device,
    )

    # --------------------------------------------------
    # Debug: fixed-subset overfit diagnostics
    # --------------------------------------------------
    debug_fixed_subset = True
    debug_indices = fixed_indices if debug_fixed_subset else [0]

    def compute_subset_debug_stats(indices):
        rows = []

        with torch.no_grad():
            for idx in indices:
                dbg_mag, dbg_rx = scene.train_set[idx]
                dbg_mag = dbg_mag.to(device).reshape(scene.beam_rows, scene.beam_cols)
                dbg_rx = dbg_rx.to(device)

                dbg_gt_mag = dbg_mag

                dbg_out = render(
                    rx_pos=dbg_rx,
                    tx_pos=tx_pos,
                    pc=gaussians,
                    rx_shape=(2, 2),
                    tx_shape=(4, 4),
                    normalize_beam_weights=False,
                    weight_floor=0.0,
                )
                dbg_pred_mag = dbg_out["render"]

                dbg_gt_shape = normalize_mag_map(dbg_gt_mag)
                dbg_pred_shape = normalize_mag_map(dbg_pred_mag)

                loss_val = magnitude_mse_loss(dbg_pred_shape, dbg_gt_shape).item()
                zero_val = torch.mean(dbg_gt_shape ** 2).item()
                ratio_val = loss_val / max(zero_val, 1e-12)

                rows.append({
                    "idx": idx,
                    "loss": loss_val,
                    "zero": zero_val,
                    "ratio_to_zero": ratio_val,
                })

        mean_loss = sum(r["loss"] for r in rows) / len(rows)
        mean_zero = sum(r["zero"] for r in rows) / len(rows)
        mean_ratio = sum(r["ratio_to_zero"] for r in rows) / len(rows)

        return rows, mean_loss, mean_zero, mean_ratio

    init_rows, init_loss, zero_loss, init_ratio = compute_subset_debug_stats(debug_indices)

    print(f"[Debug] subset mean init loss: {init_loss:.8f}")
    print(f"[Debug] subset mean zero baseline: {zero_loss:.8f}")
    print(f"[Debug] subset mean init ratio_to_zero: {init_ratio:.8f}")
    ########################################################

    num_epochs = scene.num_epochs
    total_iterations = len(scene.train_iter) * num_epochs

    print(f"[Train] Device: {device}")
    print(f"[Train] Source path: {getattr(model_params, 'source_path', '')}")
    print(f"[Train] Output path: {model_params.model_path}")
    print(f"[Train] Train set size: {len(scene.train_set)}")
    print(f"[Train] Test set size: {len(scene.test_set)}")
    print(f"[Train] Total iterations: {total_iterations}")

    iteration = 0
    ema_loss = 0.0
    progress_bar = tqdm(total = total_iterations, desc = "Training progress")

    for epoch in range(num_epochs):
        for batch in scene.train_iter:
            iteration += 1
            gaussians.update_learning_rate(iteration)

            magnitude, rx_pos = batch

            magnitude = magnitude.squeeze(0).to(device)
            rx_pos = rx_pos.squeeze(0).to(device)

            gt_mag = magnitude.reshape(scene.beam_rows, scene.beam_cols)

            assert_finite("magnitude", magnitude, iteration)
            assert_finite("rx_pos", rx_pos, iteration)

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

            assert_finite("importance", importance, iteration)

            loss, abs_loss_dbg, rel_loss_dbg, log_loss_dbg, topk_loss_dbg, bg_loss_dbg, rank_loss_dbg, gt_power_dbg = hybrid_magnitude_loss(
                pred_mag,
                gt_mag,
                alpha=0.3,
                beta=0.2,
                gamma=0.5,
                lambda_topk=0.15,
                lambda_bg=0.05,
                lambda_rank=0.10,
                topk_ratio=0.125,
                bg_threshold=0.05,
                rank_margin=0.05,
                num_neg=32,
                eps=1e-8,
                mag_scale=0.05,
                return_terms=True,
            )
            assert_finite("loss", loss, iteration)

            gaussians.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gaussians.accumulate_training_stats(importance=importance)
            gaussians.optimizer.step()


            # Densify and prune OFF FOR DEBUGGING
            # if iteration > 1000 and iteration < 15000 and iteration % 1000 == 0:
            #     with torch.no_grad():
            #         gaussians.densify_and_prune(
            #             max_grad = 1e-4,
            #             min_opacity = 1e-3,
            #             min_gain_mag = 1e-4,
            #             clone_scale_threshold=0.05,
            #             split_scale_threshold=0.20,
            #             importance_threshold=0.0,
            #             max_scale = None,
            #             n_splits = 2,
            #         )
            
            if iteration > 1000 and iteration % 1000 == 0:
                with torch.no_grad():
                    print(
                        f"grad xyz={gaussians._xyz.grad.norm().item():.3e}, "
                        f"opacity={gaussians._opacity.grad.norm().item():.3e}, "
                        f"scaling={gaussians._scaling.grad.norm().item():.3e}, "
                        f"rotation={gaussians._rotation.grad.norm().item():.3e}, "
                        f"gain_mag={gaussians._gain_mag.grad.norm().item():.3e}, "
                    )

            if iteration > 0 and iteration % 1000 == 0:
                avg_opacity = get_avg_opacity(gaussians)
                print(
                    f"nums of gaussians: {gaussians.get_xyz.shape[0]}, "
                    f"Avg opacity: {avg_opacity:.4f}, "
                    f"abs_loss: {float(abs_loss_dbg):.8f}, "
                    f"rel_loss: {float(rel_loss_dbg):.8f}, "
                    f"log_loss: {float(log_loss_dbg):.8f}, "
                    f"topk_loss: {float(topk_loss_dbg):.8f}, "
                    f"bg_loss: {float(bg_loss_dbg):.8f}, "
                    f"rank_loss: {float(rank_loss_dbg):.8f}, "
                    f"gt_power: {float(gt_power_dbg):.8f}"
                )

            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss:.8f}"
                    }
                )
                progress_bar.update(10)

    progress_bar.close()

    # --------------------------------------------------
    # Debug: fixed-subset overfit diagnostics (final)
    # --------------------------------------------------
    final_rows, final_loss, final_zero, final_ratio = compute_subset_debug_stats(debug_indices)

    print(f"[Debug] subset mean final loss: {final_loss:.8f}")
    print(f"[Debug] loss ratio final/init: {final_loss / max(init_loss, 1e-12):.8f}")
    print(f"[Debug] loss ratio final/zero: {final_loss / max(zero_loss, 1e-12):.8f}")
    print(f"[Debug] subset mean final ratio_to_zero: {final_ratio:.8f}")

    # --------------------------------------------------
    # Save per-sample debug distribution
    # --------------------------------------------------
    debug_csv_path = os.path.join(model_params.model_path, "debug_subset_losses.csv")
    with open(debug_csv_path, "w") as f:
        f.write("idx,init_loss,final_loss,zero_loss,init_ratio_to_zero,final_ratio_to_zero\n")
        for r0, rT in zip(init_rows, final_rows):
            f.write(
                f"{r0['idx']},"
                f"{r0['loss']:.8f},"
                f"{rT['loss']:.8f},"
                f"{rT['zero']:.8f},"
                f"{r0['ratio_to_zero']:.8f},"
                f"{rT['ratio_to_zero']:.8f}\n"
            )

    final_ratios = [r["ratio_to_zero"] for r in final_rows]
    final_ratios_sorted = sorted(final_ratios)

    print(f"[Debug] per-sample final ratio_to_zero min: {final_ratios_sorted[0]:.8f}")
    print(f"[Debug] per-sample final ratio_to_zero median: {final_ratios_sorted[len(final_ratios_sorted)//2]:.8f}")
    print(f"[Debug] per-sample final ratio_to_zero max: {final_ratios_sorted[-1]:.8f}")
    print(f"[Debug] saved per-sample debug csv to: {debug_csv_path}")
    ########################################################

    # --------------------------------------------------
    # Final save
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
        },
        model_ckpt,
    )
    print(f"[Save] Saved model checkpoint to {model_ckpt}")

    evaluate_and_save_random_test_samples(
        scene = scene,
        gaussians = gaussians,
        model_params = model_params,
        num_samples = 50,
    )

    print("[Train] Done.")

if __name__ == "__main__":
    parser = ArgumentParser(description="MIMOGS training script")

    model_params = ModelParams(parser)
    opt_params = OptimizationParams(parser)

    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)

    args = get_combined_args(parser)

    safe_state(args.quiet)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    mp = model_params.extract(args)
    op = opt_params.extract(args)

    training(mp, op, args)