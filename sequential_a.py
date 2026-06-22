"""
sequential_a.py

Sequential rendering / training-progress checkpointing.

Runs one normal training process over the full training dataset and saves 11
checkpoints capturing the model at 0%, 10%, 20%, ..., 100% of the total
training iterations:

    progress_000.pth  (untrained, right after init + training_setup)
    progress_010.pth
    ...
    progress_100.pth  (final trained model)

The epoch count is taken from the existing Scene / repository parameter code
(scene.num_epochs); no epoch control is added here.

Run directly:
    python sequential_a.py
"""

import os
import csv
import random
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.loss import hybrid_magnitude_loss

from train import prepare_output_dir, save_run_args_txt


SCRIPT_NAME = "sequential_a"


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

    print(f"[SequentialA] Output path: {model_params.model_path}")

    gaussians = GaussianModel(
        target_gaussians=25_000,
        optimizer_type=opt_params.optimizer_type,
        device=str(device),
        init_range=1,
    )

    scene = Scene(model_params, gaussians)

    if getattr(model_params, "init_mode", "random") == "vertices" and getattr(model_params, "vertices_path", ""):
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
        device=device,
    )

    print(f"[SequentialA] Device: {device}")
    print(f"[SequentialA] Train set size: {len(scene.train_set)}")
    print(f"[SequentialA] Num epochs: {num_epochs}")
    print(f"[SequentialA] Total iterations: {total_iterations}")

    # --------------------------------------------------
    # Checkpoint schedule: 0% (init) + 10%..100%
    # --------------------------------------------------
    progress_percents = list(range(10, 101, 10))  # 10, 20, ..., 100
    targets = {p: round(total_iterations * p / 100) for p in progress_percents}

    metadata_rows = []  # (progress_percent, checkpoint_iteration, checkpoint_path)
    saved_percents = set()

    def ckpt_path_for(percent):
        return os.path.join(ckpt_dir, f"progress_{percent:03d}.pth")

    def do_save(percent, iteration):
        path = ckpt_path_for(percent)
        save_checkpoint(
            path,
            gaussians,
            iteration,
            model_params,
            opt_params,
            args,
            scene,
            total_iterations,
        )
        saved_percents.add(percent)
        metadata_rows.append((percent, iteration, path))
        print(f"[SequentialA] Saved progress checkpoint {percent:03d}% (iter {iteration}) -> {path}")

    # progress_000: right after init + training_setup, before any optimizer update.
    do_save(0, 0)

    iteration = 0
    ema_loss = 0.0
    progress_bar = tqdm(total=total_iterations, desc="SequentialA training")

    for epoch in range(num_epochs):
        for batch in scene.train_iter:
            iteration += 1
            gaussians.update_learning_rate(iteration)

            loss = train_step(gaussians, scene, model_params, opt_params, tx_pos, batch, device)

            # Save any due checkpoints immediately after the optimizer step.
            for percent in progress_percents:
                if percent not in saved_percents and iteration >= targets[percent]:
                    do_save(percent, iteration)
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss:.8f}"
                    }
                )
                progress_bar.update(10)

    progress_bar.close()

    # Robustness: ensure every percent (including 100%) is saved even if its
    # target was never strictly reached inside the loop.
    for percent in progress_percents:
        if percent not in saved_percents:
            do_save(percent, iteration)

    # --------------------------------------------------
    # metadata.csv
    # --------------------------------------------------
    metadata_path = os.path.join(model_params.model_path, "metadata.csv")
    metadata_rows.sort(key=lambda r: r[0])
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["progress_percent", "checkpoint_iteration", "checkpoint_path"])
        for percent, it, path in metadata_rows:
            writer.writerow([percent, it, path])

    print(f"[SequentialA] Wrote metadata to {metadata_path}")
    print("[SequentialA] Done.")


if __name__ == "__main__":
    parser = ArgumentParser(description="MIMOGS sequential_a checkpointing script")

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
