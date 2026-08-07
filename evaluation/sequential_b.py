"""
sequential_b.py

Training data ratio ablation / sample efficiency experiment.

Trains separate, independent models on nested subsets of the training data
(10%, 20%, ..., 100%) and saves 11 checkpoints:

    ratio_000.pth  (common untrained initialization)
    ratio_010.pth
    ...
    ratio_100.pth

Every ratio is trained from the *same* initial Gaussian state (same Gaussian
parameters and dynamic_gain_net weights) and a fresh optimizer configured for
that ratio's total iteration count. Subsets are nested slices of one fixed
random permutation of the full training indices.

The epoch count is taken from the existing Scene / repository parameter code
(scene.num_epochs); no epoch control is added here.

Run directly:
    python sequential_b.py
"""

import os
import csv
import copy
import random
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from arguments import ModelParams, OptimizationParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.loss import hybrid_magnitude_loss

from train import prepare_output_dir, save_run_args_txt


SCRIPT_NAME = "sequential_b"


########################################################
# Shared helpers
########################################################
def make_timestamp_model_path(base_dir: str, prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{prefix}_{timestamp}")


def strip_optimizer_state(gaussian_state):
    """Drop optimizer states from a captured GaussianModel state tuple.

    Current GaussianModel.capture() layout:
        index 11 = Gaussian optimizer state
        index 12 = dynamic_gain_net state dict   (kept)
        index 13 = dynamic_gain_optimizer state
        index 14 = _xyz_tx                        (kept)
    """
    state = list(gaussian_state)
    state[11] = None  # Gaussian optimizer state
    state[13] = None  # dynamic gain optimizer state
    # index 12 (dynamic_gain_net) and index 14 (_xyz_tx) are kept untouched.
    return tuple(state)


def save_checkpoint(
    path,
    gaussians,
    iteration,
    model_params,
    opt_params,
    args,
    scene,
    total_iterations,
    extra=None,
):
    gaussian_state = gaussians.capture()
    stripped_gaussian_state = strip_optimizer_state(gaussian_state)

    ckpt = {
        "iteration": iteration,
        "gaussians": stripped_gaussian_state,
        "model_params": vars(model_params),
        "opt_params": vars(opt_params),
        "script": SCRIPT_NAME,
        "total_iterations": total_iterations,
        "num_epochs": scene.num_epochs,
        "source_path": model_params.source_path,
        "seed": args.seed,
    }
    if extra:
        ckpt.update(extra)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)
    return ckpt


########################################################
# Training step (consistent with train.py)
########################################################
def train_step(gaussians, scene, model_params, opt_params, tx_pos, batch, device):
    magnitude, rx_pos = batch

    magnitude = magnitude.squeeze(0).to(device)
    rx_pos = rx_pos.squeeze(0).to(device)

    gt_mag = magnitude.reshape(scene.beam_rows, scene.beam_cols)

    out = render(
        rx_pos=rx_pos,
        tx_pos=tx_pos,
        pc=gaussians,
        rx_shape=(2, 2),
        tx_shape=(4, 4),
        normalize_beam_weights=False,
        weight_floor=1e-4,
        max_active_rx_beams=getattr(model_params, "max_active_rx_beams", 2),
        max_active_tx_beams=getattr(model_params, "max_active_tx_beams", 2),
        renormalize_local_beam_weights=getattr(model_params, "renormalize_local_beam_weights", True),
    )
    pred_mag = out["render"]
    importance = out["per_gaussian_importance"]

    loss, abs_loss_dbg, topk_loss_dbg = hybrid_magnitude_loss(
        pred_mag,
        gt_mag,
        topk_ratio=0.0625,
        eps=1e-8,
        return_terms=True,
    )

    # Anchor-tie regularizer: pulls per-Gaussian Tx anchor toward Rx anchor.
    lambda_anchor = float(getattr(opt_params, "lambda_anchor", 1))
    anchor_reg = ((gaussians._xyz - gaussians._xyz_tx) ** 2).sum(dim=-1).mean()
    loss = loss + lambda_anchor * anchor_reg

    gaussians.optimizer.zero_grad(set_to_none=True)
    gaussians.dynamic_gain_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gaussians.accumulate_training_stats(importance=importance)
    gaussians.optimizer.step()
    gaussians.dynamic_gain_optimizer.step()

    return loss


def build_gaussians(opt_params, device, tie_covariance: bool = False):
    return GaussianModel(
        target_gaussians=25_000,
        optimizer_type=opt_params.optimizer_type,
        device=str(device),
        init_range=1,
        tie_covariance=tie_covariance,
    )


########################################################
# Main
########################################################
def run(model_params, opt_params, args):
    device = torch.device(model_params.data_device if torch.cuda.is_available() else "cpu")

    if not getattr(model_params, "model_path", None):
        model_params.model_path = make_timestamp_model_path("outputs", SCRIPT_NAME)

    prepare_output_dir(model_params.model_path)
    ckpt_dir = os.path.join(model_params.model_path, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    save_run_args_txt(model_params.model_path, model_params, opt_params, args)

    print(f"[SequentialB] Output path: {model_params.model_path}")

    # --------------------------------------------------
    # 1) Base GaussianModel + Scene (load dataset/metadata once)
    # --------------------------------------------------
    tie_covariance = bool(int(getattr(model_params, "tie_covariance", 0)))

    base_gaussians = build_gaussians(opt_params, device, tie_covariance)
    scene = Scene(model_params, base_gaussians)

    # 2) Initialize the base GaussianModel exactly like train.py.
    if getattr(model_params, "init_mode", "random") == "vertices" and getattr(model_params, "vertices_path", ""):
        base_gaussians.gaussian_init(vertices_path=model_params.vertices_path)
    else:
        base_gaussians.gaussian_init(vertices_path=None)

    # training_setup so that capture() has fully-populated statistics buffers.
    base_gaussians.training_setup(copy.deepcopy(opt_params))

    # 3) Capture the common initial state, with optimizer states stripped so that
    #    every ratio starts from identical Gaussian params + dynamic_gain_net
    #    weights and a *fresh* optimizer.
    init_state = strip_optimizer_state(base_gaussians.capture())

    tx_pos = torch.tensor(
        scene.bs_position,
        dtype=torch.float32,
        device=device,
    )

    num_epochs = scene.num_epochs
    full_train_size = len(scene.train_set)

    print(f"[SequentialB] Device: {device}")
    print(f"[SequentialB] Full train size: {full_train_size}")
    print(f"[SequentialB] Num epochs: {num_epochs}")

    # --------------------------------------------------
    # One fixed random permutation -> nested subsets.
    # --------------------------------------------------
    perm_rng = random.Random(args.seed)
    permutation = list(range(full_train_size))
    perm_rng.shuffle(permutation)

    ratio_percents = list(range(10, 101, 10))  # 10, 20, ..., 100

    metadata_rows = []  # (data_ratio_percent, num_train_samples_used, full_train_size, total_iterations, checkpoint_path)

    def ckpt_path_for(percent):
        return os.path.join(ckpt_dir, f"ratio_{percent:03d}.pth")

    # --------------------------------------------------
    # ratio_000: common untrained initialization checkpoint.
    # --------------------------------------------------
    init_path = ckpt_path_for(0)
    save_checkpoint(
        init_path,
        base_gaussians,
        0,
        model_params,
        opt_params,
        args,
        scene,
        total_iterations=0,
        extra={
            "data_ratio_percent": 0,
            "num_train_samples_used": 0,
            "full_train_size": full_train_size,
            "train_indices": [],
        },
    )
    metadata_rows.append((0, 0, full_train_size, 0, init_path))
    print(f"[SequentialB] Saved common initialization checkpoint ratio_000.pth -> {init_path}")

    # --------------------------------------------------
    # Train an independent model per nested ratio.
    # --------------------------------------------------
    for percent in ratio_percents:
        k = max(1, round(full_train_size * percent / 100))
        if percent == 100:
            k = full_train_size
        selected_indices = permutation[:k]

        subset = Subset(scene.train_set, selected_indices)
        train_loader = DataLoader(
            subset,
            batch_size=scene.batch_size,
            shuffle=True,
            num_workers=0,
        )

        total_iterations = len(train_loader) * num_epochs

        # Fresh opt_params for this ratio (never mutate the shared object).
        ratio_opt = copy.deepcopy(opt_params)
        ratio_opt.iterations = total_iterations
        ratio_opt.position_lr_max_steps = int(0.6 * total_iterations)

        # Fresh GaussianModel restored from the exact same initial state, with a
        # fresh optimizer configured for this ratio's total_iterations.
        gaussians = build_gaussians(ratio_opt, device, tie_covariance)
        gaussians.restore(init_state, ratio_opt)

        print(f"[SequentialB] Training ratio {percent:03d}% with {k} / {full_train_size} samples "
              f"({total_iterations} iterations)")

        iteration = 0
        ema_loss = 0.0
        progress_bar = tqdm(total=total_iterations, desc=f"SequentialB ratio {percent:03d}%")

        for epoch in range(num_epochs):
            for batch in train_loader:
                iteration += 1
                gaussians.update_learning_rate(iteration)

                loss = train_step(gaussians, scene, model_params, ratio_opt, tx_pos, batch, device)

                if iteration % 10 == 0:
                    ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
                    progress_bar.set_postfix(
                        {
                            "Loss": f"{ema_loss:.8f}"
                        }
                    )
                    progress_bar.update(10)


        progress_bar.close()

        path = ckpt_path_for(percent)
        save_checkpoint(
            path,
            gaussians,
            iteration,
            model_params,
            ratio_opt,
            args,
            scene,
            total_iterations=total_iterations,
            extra={
                "data_ratio_percent": percent,
                "num_train_samples_used": len(selected_indices),
                "full_train_size": full_train_size,
                "train_indices": selected_indices,
            },
        )
        metadata_rows.append((percent, len(selected_indices), full_train_size, total_iterations, path))
        print(f"[SequentialB] Saved ratio checkpoint {percent:03d}% -> {path}")

        # Free per-ratio model before the next run.
        del gaussians
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------
    # metadata.csv
    # --------------------------------------------------
    metadata_path = os.path.join(model_params.model_path, "metadata.csv")
    metadata_rows.sort(key=lambda r: r[0])
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["data_ratio_percent", "num_train_samples_used", "full_train_size", "total_iterations", "checkpoint_path"]
        )
        for row in metadata_rows:
            writer.writerow(list(row))

    print(f"[SequentialB] Wrote metadata to {metadata_path}")
    print("[SequentialB] Done.")


if __name__ == "__main__":
    parser = ArgumentParser(description="MIMOGS sequential_b data-ratio ablation script")

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

    run(mp, op, args)
