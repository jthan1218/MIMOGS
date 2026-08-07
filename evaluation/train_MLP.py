"""Pure coordinate-MLP baseline for MIMO-GS (no Gaussians, no splatting).

Maps a UE position directly to the ``(Nr, Nt)`` beam-pair power map::

    p -> PE(p) -> [Linear -> ReLU] x depth -> Linear(Nr*Nt) -> softplus

Zero-argument runnable; trains all three configurations back to back::

    python train_MLP.py

Fairness with ``train.py``
--------------------------
Everything that could bias the head-to-head is taken from the repo itself
rather than re-declared here:

* dataset + split -- built through ``scene.Scene``, which loads the prebaked
  ``train.mat`` / ``test.mat`` pair (15787 / 3947 samples, disjoint positions).
  No random splitting is involved, so the split is bit-identical to training.
* loss -- ``utils.loss.composite_magnitude_loss`` with ``topk_ratio=0.0625``,
  exactly the call ``train.py`` makes. ``train.py`` adds
  ``lambda_anchor * anchor_regularization`` on top, but ``lambda_anchor``
  defaults to 0, so the objective is identical.
* target normalization -- handled inside ``composite_magnitude_loss`` via
  ``normalize_mag_map`` (per-location max), same as MIMO-GS.
* positional encoding -- the repo's own ``FourierFeatures``, configured like
  ``DynamicGainNet`` (``num_frequencies=6``, ``include_input=True`` -> 39 dims).
* epochs / batch size / seed -- the ``ModelParams`` defaults (10 epochs,
  batch size 8) and seed 0, matching ``train.py``.

Choices that ``train.py`` does not pin down, documented here:

* output activation ``softplus`` -- the map is a power map and must be
  non-negative; softplus is what the repo already uses for the gain head.
* final-layer bias initialized to ``inverse_softplus(0.05)`` so the initial
  prediction sits near the typical normalized map level instead of at
  ``softplus(0) = 0.693``. Same trick ``DynamicGainNet.__init__`` uses.
* optimizer Adam, lr 1e-3 with per-iteration cosine annealing to 1e-5.
* ``depth`` counts hidden ``Linear+ReLU`` blocks; one output ``Linear`` is
  always appended.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from argparse import ArgumentParser
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from arguments import ModelParams, OptimizationParams
from scene import GaussianModel, Scene
from scene.gaussian_model import FourierFeatures
from utils.general_utils import inverse_softplus
from utils.loss import composite_magnitude_loss, normalize_mag_map


TOPK_RATIO = 0.0625  # identical to the train.py call
LOSS_EPS = 1e-8
INIT_OUTPUT_LEVEL = 0.05
BASE_LR = 1e-3
FINAL_LR = 1e-5
SEED = 0

CONFIGS: Dict[str, Dict[str, int]] = {
    "mlp_small": {"hidden": 256, "depth": 4},
    "mlp_medium": {"hidden": 512, "depth": 6},
    "mlp_large": {"hidden": 1024, "depth": 8},
}


class PositionMLP(nn.Module):
    """Coordinate MLP mapping a UE position to a flattened beam-pair map."""

    def __init__(
        self,
        num_outputs: int,
        hidden: int,
        depth: int,
        num_frequencies: int = 6,
        include_input: bool = True,
    ):
        super().__init__()
        self.num_outputs = int(num_outputs)
        self.hidden = int(hidden)
        self.depth = int(depth)

        self.pe = FourierFeatures(
            in_dim=3, num_frequencies=num_frequencies, include_input=include_input
        )

        layers: List[nn.Module] = []
        in_dim = self.pe.out_dim
        for _ in range(self.depth):
            layers.append(nn.Linear(in_dim, self.hidden))
            layers.append(nn.ReLU())
            in_dim = self.hidden
        output_layer = nn.Linear(in_dim, self.num_outputs)
        layers.append(output_layer)
        self.net = nn.Sequential(*layers)

        # Start near the typical normalized-map level rather than softplus(0).
        nn.init.constant_(
            output_layer.bias,
            float(inverse_softplus(torch.tensor(INIT_OUTPUT_LEVEL))),
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """``(B,3)`` positions -> ``(B, num_outputs)`` non-negative values."""
        return F.softplus(self.net(self.pe(positions)))


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def build_scene(model_params) -> Tuple[Scene, GaussianModel]:
    """Build the repo's Scene so the split/loader match train.py exactly.

    ``Scene`` requires a ``GaussianModel`` handle but never touches it during
    construction, so a one-primitive placeholder is enough.
    """
    placeholder = GaussianModel(
        target_gaussians=1,
        optimizer_type="default",
        device=str(model_params.data_device if torch.cuda.is_available() else "cpu"),
        init_range=1.0,
        tie_covariance=False,
    )
    return Scene(model_params, placeholder), placeholder


@torch.no_grad()
def evaluate_test_nmse(
    model: PositionMLP, scene: Scene, device: torch.device
) -> Tuple[float, float, float, float]:
    """Return ``(scale_dB_meanlinear, shape_dB_meanlinear, perloc_raw_dB, perloc_shape_dB)``.

    The first two follow ``train.evaluate_full_test_quality`` (average the
    linear NMSE over the test set, convert to dB once), which is the convention
    the earlier -7.72 dB MLP figure was reported in. The last two are the
    per-location-dB means that ``eval_render.py`` headlines -- ``perloc_shape``
    (normalized prediction vs normalized target) is the headline column, and
    ``perloc_raw`` is the raw-prediction convention.
    """
    model.eval()

    scale_sum = 0.0
    shape_sum = 0.0
    perloc_db_sum = 0.0
    perloc_shape_db_sum = 0.0
    count = 0

    for magnitude, rx_pos in scene.test_iter:
        magnitude = magnitude.to(device, non_blocking=True)
        rx_pos = rx_pos.to(device, non_blocking=True)

        ground_truth = magnitude.reshape(
            magnitude.shape[0], scene.beam_rows, scene.beam_cols
        )
        predicted = model(rx_pos.reshape(-1, 3)).reshape(
            -1, scene.beam_rows, scene.beam_cols
        )

        target = normalize_mag_map(ground_truth).reshape(ground_truth.shape[0], -1)
        raw = predicted.reshape(predicted.shape[0], -1)
        normalized = normalize_mag_map(predicted).reshape(predicted.shape[0], -1)

        energy = target.square().sum(dim=1).clamp_min(LOSS_EPS)
        scale_nmse = (raw - target).square().sum(dim=1) / energy
        shape_nmse = (normalized - target).square().sum(dim=1) / energy

        scale_sum += float(scale_nmse.sum().item())
        shape_sum += float(shape_nmse.sum().item())
        perloc_db_sum += float(
            (10.0 * torch.log10(scale_nmse.clamp_min(1e-12))).sum().item()
        )
        perloc_shape_db_sum += float(
            (10.0 * torch.log10(shape_nmse.clamp_min(1e-12))).sum().item()
        )
        count += int(target.shape[0])

    model.train()

    return (
        10.0 * math.log10(max(scale_sum / count, 1e-12)),
        10.0 * math.log10(max(shape_sum / count, 1e-12)),
        perloc_db_sum / count,
        perloc_shape_db_sum / count,
    )


def train_one(
    name: str,
    config: Dict[str, int],
    model_params,
    scene: Scene,
    device: torch.device,
    epochs: int,
    output_root: str,
) -> Dict[str, object]:
    """Train a single MLP configuration and save its checkpoint."""
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    num_outputs = int(scene.beam_rows * scene.beam_cols)
    model = PositionMLP(
        num_outputs=num_outputs, hidden=config["hidden"], depth=config["depth"]
    ).to(device)

    parameters = count_parameters(model)
    iterations_per_epoch = len(scene.train_iter)
    total_iterations = iterations_per_epoch * epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_iterations), eta_min=FINAL_LR
    )

    print("-" * 78)
    print(f"[train_MLP] {name}: hidden={config['hidden']} depth={config['depth']} "
          f"| params={parameters:,} | outputs={num_outputs}")
    print(f"[train_MLP] epochs={epochs} batch_size={scene.batch_size} "
          f"iters/epoch={iterations_per_epoch} total={total_iterations}")

    trajectory: List[Dict[str, float]] = []
    model.train()
    started = time.perf_counter()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_batches = 0

        for magnitude, rx_pos in scene.train_iter:
            magnitude = magnitude.to(device, non_blocking=True)
            rx_pos = rx_pos.to(device, non_blocking=True)

            ground_truth = magnitude.reshape(
                magnitude.shape[0], scene.beam_rows, scene.beam_cols
            )

            optimizer.zero_grad(set_to_none=True)
            predicted = model(rx_pos.reshape(-1, 3)).reshape(
                -1, scene.beam_rows, scene.beam_cols
            )
            loss = composite_magnitude_loss(
                predicted, ground_truth, topk_ratio=TOPK_RATIO, eps=LOSS_EPS
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.item())
            epoch_batches += 1

        scale_db, shape_db, perloc_db, perloc_shape_db = evaluate_test_nmse(
            model, scene, device
        )
        trajectory.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss / max(epoch_batches, 1),
                "test_scale_nmse_db": scale_db,
                "test_shape_nmse_db": shape_db,
                "test_perloc_mean_db": perloc_db,
                "test_perloc_shape_mean_db": perloc_shape_db,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"  epoch {epoch + 1:>2}/{epochs} | train loss {epoch_loss / max(epoch_batches, 1):.6f} "
            f"| per-loc shape {perloc_shape_db:7.3f} dB (headline) "
            f"| per-loc raw {perloc_db:7.3f} dB "
            f"| mean-linear scale {scale_db:7.3f} / shape {shape_db:7.3f} dB"
        )

    elapsed = time.perf_counter() - started

    run_dir = os.path.join(output_root, name)
    os.makedirs(run_dir, exist_ok=True)

    checkpoint = {
        "name": name,
        "state_dict": model.state_dict(),
        "config": {
            "hidden": config["hidden"],
            "depth": config["depth"],
            "num_outputs": num_outputs,
            "num_frequencies": 6,
            "include_input": True,
            "beam_rows": int(scene.beam_rows),
            "beam_cols": int(scene.beam_cols),
        },
        "training": {
            "epochs": epochs,
            "batch_size": int(scene.batch_size),
            "iterations": total_iterations,
            "optimizer": "adam",
            "lr_init": BASE_LR,
            "lr_final": FINAL_LR,
            "lr_schedule": "cosine (per iteration)",
            "loss": "composite_magnitude_loss",
            "topk_ratio": TOPK_RATIO,
            "seed": SEED,
            "source_path": str(getattr(model_params, "source_path", "")),
            "output_activation": "softplus",
            "init_output_level": INIT_OUTPUT_LEVEL,
        },
        "parameters": parameters,
        "train_seconds": elapsed,
        "trajectory": trajectory,
    }
    torch.save(checkpoint, os.path.join(run_dir, "model.pth"))

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                k: v
                for k, v in checkpoint.items()
                if k not in ("state_dict",)
            },
            handle,
            indent=2,
        )

    print(f"[train_MLP] {name} done in {elapsed:.1f} s -> {run_dir}")

    return {
        "name": name,
        "parameters": parameters,
        "run_dir": run_dir,
        "trajectory": trajectory,
        "train_seconds": elapsed,
    }


def parse_arguments() -> argparse.Namespace:
    defaults_parser = ArgumentParser()
    ModelParams(defaults_parser)
    OptimizationParams(defaults_parser)
    defaults = defaults_parser.parse_args([])

    parser = argparse.ArgumentParser(
        description="Pure coordinate-MLP baseline for MIMO-GS"
    )
    parser.add_argument("--epochs", type=int, default=int(defaults.num_epochs))
    parser.add_argument("--source_path", type=str, default=str(defaults.source_path))
    parser.add_argument("--batch_size", type=int, default=int(defaults.batch_size))
    parser.add_argument("--outputs_root", type=str, default="outputs")
    parser.add_argument(
        "--configs", type=str, default="",
        help="Comma-separated subset of: " + ",".join(CONFIGS),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    repository_root = os.path.dirname(os.path.abspath(__file__))

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    selected = (
        [c.strip() for c in arguments.configs.split(",") if c.strip()]
        if arguments.configs
        else list(CONFIGS)
    )
    unknown = [c for c in selected if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"[train_MLP] Unknown config(s): {unknown}")

    # Reuse the repo's own ModelParams so batch size / workers / array shapes
    # are resolved exactly as train.py resolves them.
    defaults_parser = ArgumentParser()
    model_group = ModelParams(defaults_parser)
    namespace = defaults_parser.parse_args([])
    namespace.source_path = arguments.source_path
    namespace.model_path = ""
    namespace.batch_size = int(arguments.batch_size)
    namespace.num_epochs = int(arguments.epochs)
    model_params = model_group.extract(namespace)

    if not os.path.isdir(model_params.source_path):
        raise SystemExit(
            f"[train_MLP] Dataset directory '{model_params.source_path}' is missing."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = arguments.outputs_root
    if not os.path.isabs(output_root):
        output_root = os.path.join(repository_root, output_root)

    scene, _placeholder = build_scene(model_params)

    print("=" * 78)
    print("[train_MLP] Pure coordinate-MLP baseline")
    print("=" * 78)
    print(f"  device      : {device}")
    print(f"  source_path : {model_params.source_path}")
    print(f"  train/test  : {len(scene.train_set)} / {len(scene.test_set)}")
    print(f"  beam grid   : {scene.beam_rows} x {scene.beam_cols}")
    print(f"  epochs      : {arguments.epochs} | batch size {scene.batch_size}")
    print(f"  configs     : {selected}")
    print("")

    results = [
        train_one(
            name, CONFIGS[name], model_params, scene, device,
            int(arguments.epochs), output_root,
        )
        for name in selected
    ]

    print("")
    print("=" * 78)
    print("[train_MLP] SUMMARY")
    print("=" * 78)
    print(f"  {'config':<14}{'params':>12}{'final scale dB':>16}{'final shape dB':>16}"
          f"{'seconds':>10}")
    for result in results:
        final = result["trajectory"][-1]
        print(
            f"  {result['name']:<14}{result['parameters']:>12,}"
            f"{final['test_scale_nmse_db']:>16.3f}{final['test_shape_nmse_db']:>16.3f}"
            f"{result['train_seconds']:>10.1f}"
        )
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
