#!/usr/bin/env python3
"""Position-MLP baseline WITHOUT positional encoding (ablation of the PE stage).

The reference row in the T1 table is ``outputs/density/MLP/model_100.pth``:
``mlp_medium`` (hidden 512, depth 6) with the repo's ``FourierFeatures``
(``num_frequencies=6``, ``include_input=True`` -> 39 input dims), trained for
50 epochs on the full ``asu_campus_16by64_lt`` train split.  This script trains
the SAME configuration with the PE stage removed (``num_frequencies=0``,
``include_input=True`` -> raw xyz, 3 input dims) and repacks it into the same
self-contained layout so ``evaluation/eval_density.load_mlp`` loads it unchanged.

Nothing in the repository is modified.  Every piece that could bias the
comparison is imported rather than re-declared:

* the model -- ``evaluation.train_MLP.PositionMLP``.  Its ``num_frequencies``
  argument is already plumbed through to ``scene.gaussian_model.FourierFeatures``,
  which handles ``num_frequencies=0`` natively (``forward`` returns ``x`` and
  ``out_dim`` is 3 when ``include_input`` is set), so no edit is needed.
  ``train_MLP.train_one`` constructs ``PositionMLP(...)`` with the default
  ``num_frequencies=6``; the module global is temporarily rebound to a factory
  that forces the override, which is why the training loop itself can be reused
  verbatim instead of copied.
* the training loop, optimizer, LR schedule, loss and checkpoint fields --
  ``evaluation.train_MLP.train_one`` (Adam, lr 1e-3 -> 1e-5 per-iteration
  cosine, ``utils.loss.composite_magnitude_loss`` with ``topk_ratio=0.0625``,
  seed 0).
* the dataset / split -- ``evaluation.train_MLP.build_scene`` -> ``scene.Scene``,
  the prebaked ``train.mat`` / ``test.mat`` pair.
* the repack layout -- mirrors
  ``evaluation.train_density_MLP.repack_mlp_checkpoint`` field for field, with
  ``arch['num_frequencies']`` read off the model that was actually trained.

Why the plain dataset directory is used instead of ``.density_tmp/mlp_frac_100``
(the ``train_source_path`` recorded in the reference checkpoint): at fraction
1.0 ``train_density.compute_keep_indices`` returns ``sort(permutation[:N])`` ==
``arange(N)``, so ``materialize_subset`` writes back the full train split in the
original order, and ``DeepMIMODataset`` casts both arrays to float32 either way.
``--verify_pe`` retrains the PE configuration here and diffs its loss trajectory
against the reference checkpoint to prove that equivalence rather than assume it.

Usage::

    python train_mlp_nope.py                    # PE off -> outputs/mlp_nope/model.pth
    python train_mlp_nope.py --verify_pe        # PE on  -> outputs/mlp_pe_repro/model.pth
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from argparse import ArgumentParser
from typing import Dict, Optional

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
EVALUATION_DIR = os.path.join(REPO_ROOT, "evaluation")
for _entry in (EVALUATION_DIR, REPO_ROOT):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from arguments import ModelParams  # noqa: E402
from evaluation import train_MLP as TM  # noqa: E402
from train_density import DATASET_EPS, dataset_scale_factor, load_train_mat  # noqa: E402


CONFIG_NAME = "mlp_medium"          # the reference row's architecture
DEFAULT_EPOCHS = 50                 # epochs recorded in model_100.pth
DEFAULT_BATCH_SIZE = 8              # ModelParams default, used by the reference
REFERENCE_CKPT = os.path.join(REPO_ROOT, "outputs", "density", "MLP", "model_100.pth")
DEFAULT_DATASET = os.path.join(REPO_ROOT, "dataset", "asu_campus_16by64_lt")


def patched_position_mlp(num_frequencies: int, include_input: bool):
    """A drop-in for ``train_MLP.PositionMLP`` that pins the PE configuration."""
    base = TM.PositionMLP

    def factory(num_outputs: int, hidden: int, depth: int, **_ignored):
        return base(
            num_outputs=num_outputs,
            hidden=hidden,
            depth=depth,
            num_frequencies=int(num_frequencies),
            include_input=bool(include_input),
        )

    return factory


def expected_input_dim(num_frequencies: int, include_input: bool) -> int:
    return 3 * ((1 if include_input else 0) + 2 * int(num_frequencies))


def repack(
    run_dir: str,
    destination: str,
    dataset_dir: str,
    n_train: int,
    normalization_scale_factor: float,
    train_seconds: float,
    epochs: int,
    num_frequencies: int,
    include_input: bool,
    note: str,
) -> Dict[str, object]:
    """Repack a train_MLP run into the density-sweep checkpoint layout.

    Field-for-field the same payload ``train_density_MLP.repack_mlp_checkpoint``
    writes, except that ``arch['num_frequencies']`` / ``arch['include_input']``
    come from this run's real PE configuration (``train_MLP.train_one`` hardcodes
    6 / True into its own ``config`` block regardless of the model it built).
    """
    raw = torch.load(os.path.join(run_dir, "model.pth"), map_location="cpu", weights_only=False)

    config = dict(raw["config"])
    hidden = int(config["hidden"])
    depth = int(config["depth"])
    num_outputs = int(config["num_outputs"])
    beam_rows = int(config["beam_rows"])
    beam_cols = int(config["beam_cols"])

    if num_outputs != beam_rows * beam_cols:
        raise AssertionError(f"[nope] num_outputs {num_outputs} != {beam_rows}x{beam_cols}")

    # The state dict is the ground truth about what PE was actually used.
    first_layer_in = int(raw["state_dict"]["net.0.weight"].shape[1])
    wanted = expected_input_dim(num_frequencies, include_input)
    if first_layer_in != wanted:
        raise AssertionError(
            f"[nope] trained first layer takes {first_layer_in} inputs but "
            f"num_frequencies={num_frequencies}/include_input={include_input} implies {wanted}"
        )

    trajectory = [float(entry["train_loss"]) for entry in raw.get("trajectory", [])]

    payload = {
        "state_dict": raw["state_dict"],
        "arch": {
            "hidden": hidden,
            "depth": depth,
            "num_frequencies": int(num_frequencies),
            "include_input": bool(include_input),
            "num_outputs": num_outputs,
        },
        "fraction": 1.0,
        "seed": TM.SEED,
        "epochs": int(epochs),
        "final_loss": trajectory[-1] if trajectory else None,
        "normalization_scale_factor": float(normalization_scale_factor),
        "beam_rows": beam_rows,
        "beam_cols": beam_cols,
        "dataset_path": os.path.abspath(dataset_dir),
        "train_source_path": str(raw.get("training", {}).get("source_path", "")),
        "n_train": int(n_train),
        "train_seconds": float(train_seconds),
        "source_run": os.path.relpath(run_dir, REPO_ROOT),
        "loss_trajectory": trajectory,
        "test_trajectory": raw.get("trajectory", []),
        "parameters": int(raw.get("parameters", 0)),
        "positional_encoding": bool(int(num_frequencies) > 0),
        "ablation_note": note,
    }

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    torch.save(payload, destination)
    print(f"[nope] repacked -> {destination}")
    return payload


def train(
    dataset_dir: str,
    output_root: str,
    destination: str,
    epochs: int,
    batch_size: int,
    num_frequencies: int,
    include_input: bool,
    note: str,
) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- train_MLP.main()'s seeding order, reproduced exactly -----------------
    random.seed(TM.SEED)
    np.random.seed(TM.SEED)
    torch.manual_seed(TM.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TM.SEED)

    # -- train_MLP.main()'s ModelParams plumbing, reproduced exactly ----------
    defaults_parser = ArgumentParser()
    model_group = ModelParams(defaults_parser)
    namespace = defaults_parser.parse_args([])
    namespace.source_path = dataset_dir
    namespace.model_path = ""
    namespace.batch_size = int(batch_size)
    namespace.num_epochs = int(epochs)
    model_params = model_group.extract(namespace)

    scene, _placeholder = TM.build_scene(model_params)

    train_positions, _train_magnitude = load_train_mat(dataset_dir)
    n_train = int(train_positions.shape[0])
    scale_factor = dataset_scale_factor(train_positions)

    print("=" * 90)
    print(f"[nope] Position-MLP ablation | num_frequencies={num_frequencies} "
          f"include_input={include_input} -> input dim "
          f"{expected_input_dim(num_frequencies, include_input)}")
    print("=" * 90)
    print(f"  device       : {device}")
    print(f"  dataset      : {dataset_dir}")
    print(f"  train/test   : {len(scene.train_set)} / {len(scene.test_set)}")
    print(f"  beam grid    : {scene.beam_rows} x {scene.beam_cols}")
    print(f"  epochs       : {epochs} | batch size {scene.batch_size} | seed {TM.SEED}")
    print(f"  config       : {CONFIG_NAME} {TM.CONFIGS[CONFIG_NAME]}")
    print(f"  scale factor : {scale_factor:.6f}")
    print("")

    original = TM.PositionMLP
    TM.PositionMLP = patched_position_mlp(num_frequencies, include_input)
    try:
        result = TM.train_one(
            CONFIG_NAME,
            TM.CONFIGS[CONFIG_NAME],
            model_params,
            scene,
            device,
            int(epochs),
            output_root,
        )
    finally:
        TM.PositionMLP = original

    payload = repack(
        run_dir=str(result["run_dir"]),
        destination=destination,
        dataset_dir=dataset_dir,
        n_train=n_train,
        normalization_scale_factor=scale_factor,
        train_seconds=float(result["train_seconds"]),
        epochs=int(epochs),
        num_frequencies=num_frequencies,
        include_input=include_input,
        note=note,
    )

    print(f"[nope] parameters   : {int(payload['parameters']):,}")
    print(f"[nope] train time   : {float(payload['train_seconds']):.1f} s")
    print(f"[nope] final loss   : {payload['final_loss']}")
    return payload


def compare_to_reference(payload: Dict[str, object]) -> None:
    """Diff a freshly trained PE run against the reference checkpoint."""
    if not os.path.isfile(REFERENCE_CKPT):
        print(f"[nope] reference checkpoint missing: {REFERENCE_CKPT}")
        return

    reference = torch.load(REFERENCE_CKPT, map_location="cpu", weights_only=False)
    a = np.asarray(reference["loss_trajectory"], dtype=np.float64)
    b = np.asarray(payload["loss_trajectory"], dtype=np.float64)
    n = min(a.size, b.size)
    delta = np.abs(a[:n] - b[:n])

    print("")
    print("[nope] PE reproduction check against outputs/density/MLP/model_100.pth")
    print(f"       epochs compared      : {n}")
    print(f"       max |loss delta|     : {delta.max():.6e}")
    print(f"       final loss ref / new : {a[-1]:.8f} / {b[-1]:.8f}")
    print(f"       parameters ref / new : {reference['parameters']} / {payload['parameters']}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Position-MLP without positional encoding")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_frequencies", type=int, default=0)
    parser.add_argument("--include_input", type=int, default=1)
    parser.add_argument(
        "--destination", type=str,
        default=os.path.join(REPO_ROOT, "outputs", "mlp_nope", "model.pth"),
    )
    parser.add_argument(
        "--run_root", type=str,
        default=os.path.join(REPO_ROOT, "outputs", "mlp_nope", "_run"),
    )
    parser.add_argument(
        "--verify_pe", action="store_true",
        help="Retrain the PE configuration (num_frequencies=6) and diff its loss "
             "trajectory against outputs/density/MLP/model_100.pth.",
    )
    return parser


def main() -> int:
    arguments = build_argument_parser().parse_args()

    dataset_dir = os.path.abspath(arguments.dataset)
    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"[nope] dataset directory is missing: {dataset_dir}")

    if arguments.verify_pe:
        payload = train(
            dataset_dir=dataset_dir,
            output_root=os.path.join(REPO_ROOT, "outputs", "mlp_pe_repro", "_run"),
            destination=os.path.join(REPO_ROOT, "outputs", "mlp_pe_repro", "model.pth"),
            epochs=int(arguments.epochs),
            batch_size=int(arguments.batch_size),
            num_frequencies=6,
            include_input=True,
            note="PE control re-run of outputs/density/MLP/model_100.pth",
        )
        compare_to_reference(payload)
        return 0

    train(
        dataset_dir=dataset_dir,
        output_root=os.path.abspath(arguments.run_root),
        destination=os.path.abspath(arguments.destination),
        epochs=int(arguments.epochs),
        batch_size=int(arguments.batch_size),
        num_frequencies=int(arguments.num_frequencies),
        include_input=bool(int(arguments.include_input)),
        note="positional encoding disabled (raw xyz input)",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
