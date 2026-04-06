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
from utils.loss import (hybrid_magnitude_loss, magnitude_mse_loss, normalize_mag_map)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter


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
                weight_floor=1e-4,
            )

            pred_mag = out["render"]

            # gt_mag_np = magnitude.detach().cpu().numpy()
            # pred_mag_np = pred_mag.detach().cpu().numpy()

            gt_mag_np = normalize_mag_map(magnitude).detach().cpu().numpy()
            pred_mag_np = pred_mag.detach().cpu().numpy()

            # fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # im0 = axes[0].imshow(gt_mag_np, aspect="equal", interpolation="nearest")
            # axes[0].set_title("Ground Truth Shape (sample-wise normalized)")
            # plt.colorbar(im0, ax=axes[0], fraction=0.03, pad=0.04)

            # im1 = axes[1].imshow(pred_mag_np, aspect="equal", interpolation="nearest")
            # axes[1].set_title("Predicted Shape (raw scale)")
            # plt.colorbar(im1, ax=axes[1], fraction=0.03, pad=0.04)

            # for ax in axes.ravel():
            #     ax.set_xlabel("Tx beam index")
            #     ax.set_ylabel("Rx beam index")
            #     ax.set_aspect("equal")

            # fig.suptitle(f"Test sample idx={idx}", fontsize=12)
            # fig.tight_layout()

            fig, axes = plt.subplots(2, 1, figsize=(8, 5), constrained_layout=True)

            # Top: Ground Truth
            im0 = axes[0].imshow(gt_mag_np, aspect="equal", interpolation="nearest")
            axes[0].set_title("Ground Truth")
            axes[0].set_xlabel("")
            axes[0].set_ylabel("")
            axes[0].set_aspect("equal")

            divider0 = make_axes_locatable(axes[0])
            cax0 = divider0.append_axes("right", size="3.5%", pad=0.08)

            cbar0 = fig.colorbar(im0, cax=cax0)
            cbar0.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            cbar0.update_ticks()

            # Bottom: Predicted
            im1 = axes[1].imshow(pred_mag_np, aspect="equal", interpolation="nearest")
            axes[1].set_title("Predicted")
            axes[1].set_xlabel("")
            axes[1].set_ylabel("")
            axes[1].set_aspect("equal")

            divider1 = make_axes_locatable(axes[1])
            cax1 = divider1.append_axes("right", size="3.5%", pad=0.08)
            cbar1 = fig.colorbar(im1, cax=cax1)
            cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            cbar1.update_ticks()
            
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
        target_gaussians = 30_000,
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

    num_epochs = scene.num_epochs
    total_iterations = len(scene.train_iter) * num_epochs
    opt_params.position_lr_max_steps = int(0.6 * total_iterations) 
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
                    weight_floor=1e-4,
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
                weight_floor=1e-4,
            )
            pred_mag = out["render"]

            importance = out["per_gaussian_importance"]

            assert_finite("importance", importance, iteration)

            loss, abs_loss_dbg, topk_loss_dbg = hybrid_magnitude_loss(
                pred_mag,
                gt_mag,
                topk_ratio=0.0625,
                eps=1e-8,
                return_terms=True,
            )
            assert_finite("loss", loss, iteration)

            gaussians.optimizer.zero_grad(set_to_none=True)
            gaussians.dynamic_gain_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gaussians.accumulate_training_stats(importance=importance)
            gaussians.optimizer.step()
            gaussians.dynamic_gain_optimizer.step()


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

                dyn_grad_norm = 0.0
                for p in gaussians.dynamic_gain_net.parameters():
                    if p.grad is not None:
                        dyn_grad_norm += p.grad.norm().item()

                with torch.no_grad():
                    print(
                        f"grad xyz={gaussians._xyz.grad.norm().item():.3e}, "
                        f"opacity={gaussians._opacity.grad.norm().item():.3e}, "
                        f"scaling={gaussians._scaling.grad.norm().item():.3e}, "
                        f"rotation={gaussians._rotation.grad.norm().item():.3e}, "
                        # f"gain_mag={gaussians._gain_mag.grad.norm().item():.3e}, "
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