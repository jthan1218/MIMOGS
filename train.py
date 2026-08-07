from __future__ import annotations

import math
import os
import random
import time
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, get_combined_args
from gaussian_renderer.fast_renderer import render_fast
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.loss import composite_magnitude_loss, normalize_mag_map


# Post-training evaluation settings.
# Change these two values directly; no additional CLI options are required.
NUM_EVAL_SAMPLES = 50
EVAL_BATCH_SIZE = 50


def make_output_path(base_dir: str = "outputs") -> str:
    """Create a timestamped output directory path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, timestamp)


def prepare_output_dir(model_path: str) -> None:
    """Create output subdirectories."""
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(os.path.join(model_path, "point_cloud"), exist_ok=True)
    os.makedirs(os.path.join(model_path, "pred_compare"), exist_ok=True)


def save_args(path: str,model_params,opt_params,raw_args) -> None:
    """Save all command-line and model parameters."""
    os.makedirs(path, exist_ok=True)

    args_path = os.path.join(path, "run_args.txt")

    with open(args_path, "w", encoding="utf-8") as file:
        for title, obj in [("Model Params", model_params),("Optimization Params", opt_params),("Raw Args", raw_args)]:
            file.write(f"[{title}]\n")

            for key, value in sorted(vars(obj).items()):
                file.write(f"{key}: {value}\n")

            file.write("\n")


def evaluate_and_save_random_test_samples(
    scene: Scene,
    gaussians: GaussianModel,
    model_params,
) -> None:
    """Render and save comparison figures for random test samples."""

    save_dir = os.path.join(model_params.model_path,"pred_compare")
    os.makedirs(save_dir, exist_ok=True)

    total_test_samples = len(scene.test_set)

    if total_test_samples == 0:
        print("[Evaluation] Test set is empty. Skipping figure generation.")
        return

    num_samples = min(NUM_EVAL_SAMPLES,total_test_samples)
    eval_batch_size = min(max(1, int(EVAL_BATCH_SIZE)),num_samples)

    # Fixed seed ensures that the same test samples are selected every run.
    random_generator = random.Random(12345)
    selected_indices = random_generator.sample(range(total_test_samples),num_samples)

    # Load the selected samples on CPU first, then transfer each stacked
    # tensor to the model device only once.
    selected_samples = [scene.test_set[test_index] for test_index in selected_indices]
    ground_truth_batch_cpu = torch.stack([magnitude.reshape(scene.beam_rows,scene.beam_cols) for magnitude, _ in selected_samples],dim=0)
    rx_pos_batch_cpu = torch.stack([rx_pos.reshape(3) for _, rx_pos in selected_samples],dim=0)

    device = gaussians.get_xyz.device

    ground_truth_batch = ground_truth_batch_cpu.to(device,non_blocking=True)

    rx_pos_batch = rx_pos_batch_cpu.to(device,non_blocking=True)
    tx_pos = torch.as_tensor(scene.bs_position,dtype=torch.float32,device=device)

    print(f"[Evaluation] Rendering {num_samples} random test samples with eval batch size {eval_batch_size}...")

    gaussians.dynamic_gain_net.eval()

    with torch.inference_mode():
        # Ensure that input transfers are complete before starting the render
        # timer. CUDA operations are asynchronous.
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        render_start = time.perf_counter()
        predicted_chunks = []

        for start_index in range(0,num_samples,eval_batch_size):
            end_index = min(start_index + eval_batch_size,num_samples)

            rendered_output = render_fast(
                rx_pos=rx_pos_batch[start_index:end_index],
                tx_pos=tx_pos,
                pc=gaussians,
                rx_shape=scene.rx_shape,
                tx_shape=scene.tx_shape,
                covariance_floor=1e-4,
                weight_floor=1e-4,
                max_active_rx_beams=int(getattr(model_params, "max_active_rx_beams", 2)),
                max_active_tx_beams=int(getattr(model_params, "max_active_tx_beams", 2)),
                use_cuda_rasterizer=bool(int(getattr(model_params, "use_cuda_rasterizer", 1))),
                beam_grid_mode=scene.beam_grid_mode,
                beam_az_deg=scene.beam_az_deg,
                beam_el_deg=scene.beam_el_deg,
            )

            predicted_chunk = rendered_output["render"]

            # Keep a batch dimension even when a chunk contains one sample.
            if predicted_chunk.ndim == 2:
                predicted_chunk = predicted_chunk.unsqueeze(0)

            predicted_chunks.append(predicted_chunk)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        render_time = time.perf_counter() - render_start

        # All rendering is complete before any result is transferred to CPU.
        predicted_batch = torch.cat(predicted_chunks,dim=0)

        if predicted_batch.shape[0] != num_samples:
            raise RuntimeError(f"The number of rendered predictions does not match the requested sample count: {predicted_batch.shape[0]} != {num_samples}")

        ground_truth_normalized = normalize_mag_map(ground_truth_batch)

        # Exclude concatenation and normalization from transfer timing.
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        predicted_batch_cpu = predicted_batch.detach().cpu()

        ground_truth_batch_cpu = ground_truth_normalized.detach().cpu()

        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # NumPy conversion happens after the measured GPU-to-CPU transfer.
    predicted_numpy = predicted_batch_cpu.numpy()
    ground_truth_numpy = ground_truth_batch_cpu.numpy()

    plot_and_save_start = time.perf_counter()

    for output_index in range(num_samples):
        ground_truth_map_numpy = ground_truth_numpy[output_index]
        predicted_map_numpy = predicted_numpy[output_index]

        # if dataset is in power domain, convert to dB scale
        # gt_db   = 10.0 * np.log10(ground_truth_map_numpy + 1e-12)
        # pred_db = 10.0 * np.log10(np.clip(predicted_map_numpy, 0, None) + 1e-12)

        figure, axes = plt.subplots(
            2,
            1,
            figsize=(8, 5),
            constrained_layout=True,
        )

        # Ground-truth map
        ground_truth_image = axes[0].imshow(ground_truth_map_numpy,aspect="equal",interpolation="nearest")
        
        # use dB scale for ground truth (power dataset case)
        # ground_truth_image = axes[0].imshow(gt_db, aspect="equal", interpolation="nearest",vmin=-50, vmax=0)
        axes[0].set_title("Ground Truth")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("")
        axes[0].set_aspect("equal")

        ground_truth_divider = make_axes_locatable(axes[0])
        ground_truth_colorbar_axis = ground_truth_divider.append_axes("right",size="3.5%",pad=0.08)
        ground_truth_colorbar = figure.colorbar(ground_truth_image,cax=ground_truth_colorbar_axis)
        ground_truth_colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ground_truth_colorbar.update_ticks()

        # Predicted map
        predicted_image = axes[1].imshow(predicted_map_numpy,aspect="equal",interpolation="nearest")

        # use dB scale for predicted (power dataset case)
        # predicted_image = axes[1].imshow(pred_db, aspect="equal", interpolation="nearest",vmin=-50, vmax=0)

        axes[1].set_title("Predicted")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("")
        axes[1].set_aspect("equal")

        predicted_divider = make_axes_locatable(axes[1])

        predicted_colorbar_axis = predicted_divider.append_axes("right",size="3.5%",pad=0.08)
        predicted_colorbar = figure.colorbar(predicted_image,cax=predicted_colorbar_axis)
        predicted_colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        predicted_colorbar.update_ticks()

        figure_path = os.path.join(save_dir,f"{output_index:02d}.png")
        figure.savefig(figure_path,dpi=150)

        plt.close(figure)

    plot_and_save_time = time.perf_counter() - plot_and_save_start

    gaussians.dynamic_gain_net.train()

    print(f"[Evaluation] Saved comparison figures to {save_dir}")
    print(f"[Evaluation][Timing] render: {render_time:.4f} s")
    print(f"[Evaluation][Timing] plot_and_save: {plot_and_save_time:.4f} s")


def evaluate_full_test_quality(
    scene: Scene,
    gaussians: GaussianModel,
    model_params,
) -> None:
    """Render the entire test set in batches and print scale/shape NMSE.

    scale NMSE compares the raw prediction with the max-normalized ground
    truth; shape NMSE compares the max-normalized prediction with the
    max-normalized ground truth. Per-sample linear NMSE values are averaged
    over the full test set first and the mean is converted to dB once.
    No figures are saved here.
    """

    total_test_samples = len(scene.test_set)

    if total_test_samples == 0:
        print("[Evaluation][Quality] Test set is empty. Skipping quality evaluation.")
        return

    device = gaussians.get_xyz.device
    tx_pos = torch.as_tensor(scene.bs_position,dtype=torch.float32,device=device)

    gaussians.dynamic_gain_net.eval()

    scale_nmse_sum = 0.0
    shape_nmse_sum = 0.0
    evaluated_samples = 0

    with torch.inference_mode():
        for magnitude, rx_pos in scene.test_iter:
            magnitude = magnitude.to(device,non_blocking=True)
            rx_pos = rx_pos.to(device,non_blocking=True)

            ground_truth_map = magnitude.reshape(magnitude.shape[0],scene.beam_rows,scene.beam_cols)

            rendered_output = render_fast(
                rx_pos=rx_pos.reshape(-1, 3),
                tx_pos=tx_pos,
                pc=gaussians,
                rx_shape=scene.rx_shape,
                tx_shape=scene.tx_shape,
                covariance_floor=1e-4,
                weight_floor=1e-4,
                max_active_rx_beams=int(getattr(model_params, "max_active_rx_beams", 2)),
                max_active_tx_beams=int(getattr(model_params, "max_active_tx_beams", 2)),
                use_cuda_rasterizer=bool(int(getattr(model_params, "use_cuda_rasterizer", 1))),
                beam_grid_mode=scene.beam_grid_mode,
                beam_az_deg=scene.beam_az_deg,
                beam_el_deg=scene.beam_el_deg,
            )

            predicted_map = rendered_output["render"]

            if predicted_map.ndim == 2:
                predicted_map = predicted_map.unsqueeze(0)

            ground_truth_normalized = normalize_mag_map(ground_truth_map)
            predicted_normalized = normalize_mag_map(predicted_map)

            target_flat = ground_truth_normalized.reshape(ground_truth_normalized.shape[0],-1)
            raw_predicted_flat = predicted_map.reshape(predicted_map.shape[0],-1)
            normalized_predicted_flat = predicted_normalized.reshape(predicted_normalized.shape[0],-1)

            target_energy = target_flat.square().sum(dim=1).clamp_min(1e-8)
            scale_nmse = (raw_predicted_flat - target_flat).square().sum(dim=1) / target_energy
            shape_nmse = (normalized_predicted_flat - target_flat).square().sum(dim=1) / target_energy

            scale_nmse_sum += float(scale_nmse.sum().item())
            shape_nmse_sum += float(shape_nmse.sum().item())
            evaluated_samples += int(target_flat.shape[0])

    gaussians.dynamic_gain_net.train()

    mean_scale_nmse = scale_nmse_sum / evaluated_samples
    mean_shape_nmse = shape_nmse_sum / evaluated_samples

    scale_nmse_db = 10.0 * math.log10(max(mean_scale_nmse, 1e-12))
    shape_nmse_db = 10.0 * math.log10(max(mean_shape_nmse, 1e-12))

    print(f"[Evaluation][Quality] N={evaluated_samples} | scale NMSE: {scale_nmse_db:.2f} dB | shape NMSE: {shape_nmse_db:.2f} dB")


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

def training(
    model_params,
    opt_params,
    raw_args,
) -> None:
    """Run batched MIMO-GS training."""

    device = torch.device(model_params.data_device if torch.cuda.is_available() else "cpu")

    if not getattr(model_params, "model_path", None):
        model_params.model_path = make_output_path()

    prepare_output_dir(model_params.model_path)

    save_args(model_params.model_path,model_params,opt_params,raw_args)

    gaussians = GaussianModel(target_gaussians=int(getattr(model_params, "target_gaussians", 25_000)),optimizer_type=opt_params.optimizer_type,device=str(device),init_range=1.0,tie_covariance=bool(int(getattr(model_params, "tie_covariance", 0))))

    scene = Scene(model_params,gaussians)

    if getattr(model_params, "init_mode", "random") == "vertices" and getattr(model_params, "vertices_path", ""):
        vertices_path = model_params.vertices_path
    else:
        vertices_path = None

    gaussians.gaussian_init(vertices_path=vertices_path)

    total_iterations = len(scene.train_iter)* scene.num_epochs

    if opt_params.iterations <= 0:
        opt_params.iterations = total_iterations

    if opt_params.position_lr_max_steps <= 0:
        opt_params.position_lr_max_steps = max(1,int(0.6 * total_iterations))

    gaussians.training_setup(opt_params)

    tx_pos = torch.as_tensor(scene.bs_position,dtype=torch.float32,device=device)

    use_amp = bool(int(getattr(model_params, "use_amp", 0))) and device.type == "cuda"

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"[Train] Device: {device} | Source path: {getattr(model_params, 'source_path', '')} | Output path: {model_params.model_path}")

    print(f"[Train] Train set size: {len(scene.train_set)} | Test set size: {len(scene.test_set)} | Batch size: {getattr(model_params, 'batch_size', 'unknown')} | Total iterations: {total_iterations} | Epochs: {scene.num_epochs}")

    if scene.beam_grid_mode == "custom_angles":
        print(f"[Train] Beam grid: custom_angles, {len(scene.beam_az_deg)} az x {len(scene.beam_el_deg)} el = {scene.beam_rows} beams (non-periodic, wrap disabled)")
    else:
        print(f"[Train] Beam grid: Rx {scene.rx_shape} = {scene.beam_rows} beams | Tx {scene.tx_shape} = {scene.beam_cols} beams")

    iteration = 0
    ema_loss = 0.0

    progress_bar = tqdm(total=total_iterations,desc="MIMO-GS training")

    gaussians.dynamic_gain_net.train()

    for epoch in range(scene.num_epochs):
        for magnitude, rx_pos in scene.train_iter:
            iteration += 1

            gaussians.update_learning_rate(iteration)
            magnitude = magnitude.to(device,non_blocking=True)
            rx_pos = rx_pos.to(device,non_blocking=True)
            ground_truth_map = magnitude.reshape(magnitude.shape[0],scene.beam_rows,scene.beam_cols)
            gaussians.optimizer.zero_grad(set_to_none=True)
            gaussians.dynamic_gain_optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(
                enabled=use_amp
            ):
                rendered_output = render_fast(
                    rx_pos=rx_pos,
                    tx_pos=tx_pos,
                    pc=gaussians,
                    rx_shape=scene.rx_shape,
                    tx_shape=scene.tx_shape,
                    covariance_floor=1e-4,
                    weight_floor=1e-4,
                    max_active_rx_beams=int(
                        model_params.max_active_rx_beams
                    ),
                    max_active_tx_beams=int(
                        model_params.max_active_tx_beams
                    ),
                    use_cuda_rasterizer=bool(
                        int(
                            model_params.use_cuda_rasterizer
                        )
                    ),
                    beam_grid_mode=scene.beam_grid_mode,
                    beam_az_deg=scene.beam_az_deg,
                    beam_el_deg=scene.beam_el_deg,
                )

                predicted_map = rendered_output["render"]

                reconstruction_loss,scale_term,shape_term,topk_term = composite_magnitude_loss(
                    predicted_map,
                    ground_truth_map,
                    topk_ratio=0.0625,
                    eps=1e-8,return_terms=True)

                lambda_anchor = float(
                    getattr(
                        opt_params,
                        "lambda_anchor",
                        1.0,
                    )
                )

                anchor_regularization = (gaussians._xyz - gaussians._xyz_tx).square().sum(dim=-1).mean()

                loss = reconstruction_loss+ lambda_anchor* anchor_regularization

                importance = rendered_output["per_gaussian_importance"].mean(dim=0)

            scaler.scale(loss).backward()
            scaler.unscale_(gaussians.optimizer)
            gaussians.accumulate_training_stats(importance=importance)

            scaler.step(gaussians.optimizer)
            scaler.step(gaussians.dynamic_gain_optimizer)
            scaler.update()

            # if (
            #     iteration > 1000
            #     and iteration < 15000
            #     and iteration % 1000 == 0
            # ):
            #     with torch.no_grad():
            #         gaussians.densify_and_prune(
            #             max_grad=1e-4,
            #             min_opacity=1e-3,
            #             clone_scale_threshold=0.05,
            #             split_scale_threshold=0.20,
            #             importance_threshold=0.0,
            #             max_scale=None,
            #             n_splits=2,
            #         )

            ema_loss = (0.4 * loss.item()+ 0.6 * ema_loss)

            if iteration > 0 and iteration % 1000 == 0:
                avg_opacity = get_avg_opacity(gaussians)
                print(
                    f"nums of gaussians: {gaussians.get_xyz.shape[0]}, "
                    f"Avg opacity: {avg_opacity:.4f}, "
                )

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss:.8f}"
                    }
                )
                progress_bar.update(10)

    progress_bar.close()

    # Save Gaussian point cloud.
    point_cloud_path = os.path.join(model_params.model_path,"point_cloud","point_cloud.ply")

    if hasattr(gaussians, "save_ply"):
        gaussians.save_ply(
            point_cloud_path
        )

        print(f"[Save] Saved point cloud to {point_cloud_path}")

    # Save checkpoint.
    checkpoint_path = os.path.join(model_params.model_path,"model.pth")

    torch.save({"iteration": iteration,"gaussians": gaussians.capture(),"model_params": vars(model_params),"opt_params": vars(opt_params)},checkpoint_path)

    print(f"[Save] Saved checkpoint to {checkpoint_path}")

    # Save random test rendering results using the fixed settings above.
    evaluate_and_save_random_test_samples(scene,gaussians,model_params)

    # Full-test-set quality metrics (printed only; no extra images).
    evaluate_full_test_quality(scene,gaussians,model_params)

    print("[Train] Done.")


if __name__ == "__main__":
    parser = ArgumentParser(description="MIMO-GS training")

    model_params = ModelParams(parser)
    optimization_params = OptimizationParams(parser)

    parser.add_argument("--quiet",action="store_true",default=False)
    parser.add_argument("--seed",type=int,default=0)

    args = get_combined_args(parser)

    safe_state(args.quiet)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    extracted_model_params = model_params.extract(args)
    extracted_optimization_params = optimization_params.extract(args)

    training(extracted_model_params,extracted_optimization_params,args)
