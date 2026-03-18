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


def complex_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target) ** 2)


def wrapped_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


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
    indices = random.sample(range(total), num_samples)

    tx_pos = torch.tensor(
        scene.bs_position,
        dtype=torch.float32,
        device=gaussians.get_xyz.device,
    )

    print(f"[Evaluation] Rendering {num_samples} random test samples...")

    with torch.no_grad():
        for rank, idx in enumerate(indices):
            magnitude, phases, rx_pos = scene.test_set[idx]

            rx_pos = rx_pos.to(gaussians.get_xyz.device)
            magnitude = magnitude.to(gaussians.get_xyz.device)
            phases = phases.to(gaussians.get_xyz.device)

            magnitude = magnitude.reshape(scene.beam_rows, scene.beam_cols)
            phases = phases.reshape(scene.beam_rows, scene.beam_cols)

            out = render(
                rx_pos = rx_pos,
                tx_pos = tx_pos,
                pc = gaussians,
                rx_shape = (2, 2),
                tx_shape = (4, 4),
                use_geometric_phase=model_params.use_geometric_phase,
                carrier_frequency_hz = model_params.carrier_frequency_hz,
            )

            pred_H = out["render"]
            pred_mag = torch.abs(pred_H)
            pred_phase = torch.angle(pred_H)

            gt_mag_np = magnitude.detach().cpu().numpy()
            pred_mag_np = pred_mag.detach().cpu().numpy()

            # GT phase is assumed to be in [0, 2pi), but pred phase is from angle() in (-pi, pi]
            # Wrap GT for visually consistent comparison
            gt_phase_wrapped = wrapped_to_pi(phases).detach().cpu().numpy()
            pred_phase_np = pred_phase.detach().cpu().numpy()

            fig, axes = plt.subplots(2, 2, figsize=(10, 5))

            im0 = axes[0, 0].imshow(gt_mag_np, aspect="auto")
            axes[0, 0].set_title("Ground Truth Magnitude")
            plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

            im1 = axes[0, 1].imshow(pred_mag_np, aspect="auto")
            axes[0, 1].set_title("Predicted Magnitude")
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

            im2 = axes[1, 0].imshow(gt_phase_wrapped, aspect="auto", vmin=-np.pi, vmax=np.pi)
            axes[1, 0].set_title("Ground Truth Phase")
            plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

            im3 = axes[1, 1].imshow(pred_phase_np, aspect="auto", vmin=-np.pi, vmax=np.pi)
            axes[1, 1].set_title("Predicted Phase")
            plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

            for ax in axes.ravel():
                ax.set_xlabel("Tx beam index")
                ax.set_ylabel("Rx beam index")

            fig.suptitle(f"Test sample idx={idx}", fontsize=12)
            fig.tight_layout()

            fig_path = os.path.join(save_dir, f"{rank:02d}.png")
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)

    print(f"[Eval] Saved comparison figures to {save_dir}")

def training(model_params, opt_params, raw_args):
    device = torch.device(model_params.data_device if torch.cuda.is_available() else "cpu")

    if not getattr(model_params, "model_path", None):
        model_params.model_path = make_timestamp_model_path("outputs")
    
    prepare_output_dir(model_params.model_path)
    save_run_args_txt(model_params.model_path, model_params, opt_params, raw_args)

    gaussians = GaussianModel(
        target_gaussians = 50_000,
        optimizer_type = opt_params.optimizer_type,
        device = str(device),
        init_range = 5.0,
    )

    scene = Scene(model_params, gaussians)

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

            magnitude, phases, rx_pos = batch

            magnitude = magnitude.squeeze(0).to(device)
            phases = phases.squeeze(0).to(device)
            rx_pos = rx_pos.squeeze(0).to(device)

            magnitude = magnitude.reshape(scene.beam_rows, scene.beam_cols)
            phases = phases.reshape(scene.beam_rows, scene.beam_cols)

            gt_H = torch.polar(magnitude, phases)

            out = render(
                rx_pos = rx_pos,
                tx_pos = tx_pos,
                pc = gaussians,
                rx_shape = (2, 2),
                tx_shape = (4, 4),
                use_geometric_phase=model_params.use_geometric_phase,
                carrier_frequency_hz = model_params.carrier_frequency_hz,
            )

            pred_H = out["render"]
            importance = out["per_gaussian_importance"]

            loss = complex_mse_loss(pred_H, gt_H)

            gaussians.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gaussians.accumulate_training_stats(importance=importance)
            gaussians.optimizer.step()


            if iteration > 1000 and iteration < 15000 and iteration % 1000 == 0:
                with torch.no_grad():
                    gaussians.densify_and_prune(
                        max_grad = 1e-4,
                        min_opacity = 1e-3,
                        min_gain_mag = 1e-4,
                        clone_scale_threshold=0.05,
                        split_scale_threshold=0.20,
                        importance_threshold=0.0,
                        max_scale = None,
                        n_splits = 2,
                    )

            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            progress_bar.set_postfix(
                {
                    "Loss": f"{ema_loss:.6e}",
                    "Num Gaussians": int(gaussians.get_xyz.shape[0]),
                }
            )
            progress_bar.update(1)

    progress_bar.close()

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