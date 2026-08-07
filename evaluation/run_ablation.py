"""E3 -- ablation study runner for MIMO-GS.

Runs with zero arguments::

    python run_ablation.py

Each training variant is launched as a ``python train.py --<flags>`` subprocess
with its own ``--model_path outputs/ablation_<name>``, so runs never collide.
Every trained variant is then evaluated on the same test set with the same
metric definitions as :mod:`eval_render` (per-location max-normalized target,
see ``utils/loss.py::normalize_mag_map`` and the ``composite_magnitude_loss``
scale term used in ``train.py``).

Variants
--------
(a) ``baseline``          -- repo defaults.
(b) ``tied_anchor``       -- ``--lambda_anchor <large>`` pins the Tx anchor to
    the Rx anchor. ``train.py`` builds the penalty as
    ``(_xyz - _xyz_tx).square().sum(-1).mean()``, i.e. the mean over ALL
    Gaussians, so the per-Gaussian gradient carries a 1/N factor with
    N = target_gaussians. The multiplier therefore has to be several orders of
    magnitude above 1 to actually pin the anchors; the resulting mean
    ``||q_r - q_t||`` is measured from the checkpoint and reported.
(c) ``tied_covariance``   -- ``--tie_covariance 1``.
(d) ``static_gain``       -- ``--dynamic_gain_lr 0 --dynamic_gain_lr_final 0``.
    ``DynamicGainNet.__init__`` zeroes the last layer's weight and sets its
    bias to ``inverse_softplus(0.1)``, so at initialization the network output
    is input-independent: ``d_k(p) == 0.1`` for every Gaussian and every UE
    position. ``get_expon_lr_func`` returns exactly 0.0 when both endpoints are
    zero, so the network stays frozen and the location-conditioned gain is
    disabled, leaving only the static per-Gaussian opacity. No code change is
    needed; the check in ``verify_static_gain`` confirms it empirically.
(e) ``topk_sweep``        -- eval-time only, no retraining: the baseline
    checkpoint is re-rendered with several ``max_active_rx/tx_beams`` settings.

Outputs land in ``analysis/ablation/``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import subprocess
import sys
import time
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from arguments import ModelParams, OptimizationParams

try:
    import eval_render
except ImportError as error:  # pragma: no cover - eval_render ships with E1.
    raise SystemExit(
        "[run_ablation] eval_render.py is required for the metric definitions "
        f"(NMSE / top-K overlap / power capture) but could not be imported: {error}"
    )


# The fused CUDA rasterizer stores its top-k in a fixed-size register array;
# see ``kMaxTopK`` in mimogs_rasterizer/csrc/rasterizer_cuda.cu. Larger K must
# fall back to the differentiable PyTorch reference path.
CUDA_MAX_TOPK = 8

# ``train.py`` averages the anchor penalty over all Gaussians, so the effective
# per-Gaussian pull is lambda/N. This value is large enough to pin the anchors
# for the default N = 25_000; OptimizationParams declares lambda_anchor as an
# int, so it must stay integral on the command line.
TIED_ANCHOR_LAMBDA = 10_000

REPORT_TOPK = (1, 4, 8)
CAPTURE_K = 4


# ----------------------------------------------------------------------
# Variant definitions
# ----------------------------------------------------------------------
TRAINING_VARIANTS: List[Dict[str, object]] = [
    {
        "key": "baseline",
        "letter": "a",
        "label": "(a) baseline",
        "flags": [],
        "description": "repo default settings",
    },
    {
        "key": "tied_anchor",
        "letter": "b",
        "label": "(b) tied anchor",
        "flags": ["--lambda_anchor", str(TIED_ANCHOR_LAMBDA)],
        "description": f"anchor-tie regularizer lambda_anchor={TIED_ANCHOR_LAMBDA}",
    },
    {
        "key": "tied_covariance",
        "letter": "c",
        "label": "(c) tied covariance",
        "flags": ["--tie_covariance", "1"],
        "description": "single (scaling, rotation) pair shared by both sides",
    },
    {
        "key": "static_gain",
        "letter": "d",
        "label": "(d) static gain",
        "flags": ["--dynamic_gain_lr", "0", "--dynamic_gain_lr_final", "0"],
        "description": "gain MLP frozen at its constant init -> d_k(p) == 0.1",
    },
]

SWEEP_KEY = "topk_sweep"
SWEEP_LETTER = "e"


def repo_default(group_cls, attribute: str, fallback):
    """Read a default straight out of the repo's argument definitions."""
    parser = ArgumentParser()
    group_cls(parser)
    return getattr(parser.parse_args([]), attribute, fallback)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def build_training_command(
    variant: Dict[str, object],
    model_path: str,
    epochs: int,
    source_path: str,
    target_gaussians: int,
) -> List[str]:
    command = [
        sys.executable,
        "train.py",
        "--source_path",
        source_path,
        "--model_path",
        model_path,
        "--num_epochs",
        str(int(epochs)),
        "--target_gaussians",
        str(int(target_gaussians)),
    ]
    command += [str(flag) for flag in variant["flags"]]
    return command


def run_training(
    command: Sequence[str], repository_root: str, log_path: str
) -> Tuple[bool, float, str]:
    """Launch train.py; never raise, so remaining variants still run."""
    started = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        try:
            completed = subprocess.run(
                list(command),
                cwd=repository_root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        except OSError as error:
            return False, time.perf_counter() - started, f"could not launch: {error}"

    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        return (
            False,
            elapsed,
            f"train.py exited with code {completed.returncode} (see {log_path})",
        )
    return True, elapsed, ""


# ----------------------------------------------------------------------
# Checkpoint loading / diagnostics
# ----------------------------------------------------------------------
def load_trained_run(run_dir: str, device: torch.device):
    """Restore a run exactly the way eval_render does."""
    checkpoint_path = os.path.join(run_dir, eval_render.CHECKPOINT_NAME)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_params, opt_params = eval_render.restore_config(run_dir, checkpoint)
    scene, gaussians = eval_render.build_scene_and_model(
        model_params, opt_params, checkpoint, device
    )
    return scene, gaussians, model_params, opt_params, checkpoint


def mean_anchor_distance(gaussians) -> float:
    """Mean ``||q_r - q_t||`` over the Gaussians, in normalized scene units."""
    with torch.no_grad():
        separation = gaussians.get_xyz - gaussians.get_xyz_tx
        return float(separation.norm(dim=-1).mean().item())


def verify_static_gain(gaussians, scene, device, num_probes: int = 32) -> float:
    """Relative spread of the per-Gaussian gain across UE positions.

    Exactly 0.0 means the gain network ignores its input, i.e. the
    location-conditioned gain ``d_k(p)`` is genuinely disabled.
    """
    total = len(scene.test_set)
    probes = min(int(num_probes), total)
    stride = max(1, total // probes)
    positions = torch.stack(
        [scene.test_set[i][1].reshape(3) for i in range(0, stride * probes, stride)],
        dim=0,
    ).to(device)

    with torch.no_grad():
        gains = gaussians.get_dynamic_gain_weight_batched(positions)
        spread = gains.std(dim=0) / gains.mean(dim=0).abs().clamp_min(eval_render.EPS)
        return float(spread.mean().item())


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def params_with_beam_limits(model_params: Namespace, k_rx: int, k_tx: int) -> Namespace:
    """Copy the restored config with the renderer's top-k overridden.

    ``max_active_rx_beams`` / ``max_active_tx_beams`` are baked into the saved
    config, so the sweep overrides them on the copy that is handed to
    ``render_fast`` instead of touching the checkpoint.
    """
    overridden = copy.copy(model_params)
    overridden.max_active_rx_beams = int(k_rx)
    overridden.max_active_tx_beams = int(k_tx)
    return overridden


def backend_for(k_rx: int, k_tx: int, cuda_available: bool) -> Tuple[bool, str]:
    """Pick the rasterizer backend that can actually serve this K."""
    if not cuda_available:
        return False, "reference-cpu"
    if max(int(k_rx), int(k_tx)) > CUDA_MAX_TOPK:
        return False, "reference-fallback"
    return True, "cuda"


def collect_metrics(results: Dict) -> Dict[str, float]:
    nmse = results["nmse_raw_db"]
    metrics: Dict[str, float] = {
        "nmse_mean_dB": float(np.mean(nmse)),
        "nmse_median_dB": float(np.median(nmse)),
        "nmse_meanlinear_dB": eval_render.mean_linear_db(nmse),
        "nmse_shape_mean_dB": float(np.mean(results["nmse_shape_db"])),
        "num_evaluated": int(results["index"].shape[0]),
    }
    for k in REPORT_TOPK:
        metrics[f"topk_acc_K{k}"] = float(np.mean(results["topk"][k]))
    metrics[f"power_capture_K{CAPTURE_K}"] = float(
        np.mean(results["capture"][CAPTURE_K])
    )
    return metrics


def measure_render_time(
    scene,
    gaussians,
    model_params: Namespace,
    device: torch.device,
    batch_size: int,
    use_cuda_rasterizer: bool,
) -> float:
    """Milliseconds of render_fast wall clock per map, warmup excluded.

    Positions are moved to the device up front so the measurement covers the
    renderer only, not the data pipeline.
    """
    positions = getattr(scene.test_set, "positions", None)
    if positions is None:
        positions = torch.stack(
            [scene.test_set[i][1].reshape(3) for i in range(len(scene.test_set))], dim=0
        )
    positions = positions.reshape(-1, 3).to(device)

    tx_pos = torch.as_tensor(scene.bs_position, dtype=torch.float32, device=device)
    total = int(positions.shape[0])

    with torch.no_grad():
        eval_render.render_batch(
            positions[: min(batch_size, total)],
            tx_pos,
            gaussians,
            scene,
            model_params,
            use_cuda_rasterizer,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        started = time.perf_counter()
        for start in range(0, total, batch_size):
            eval_render.render_batch(
                positions[start : start + batch_size],
                tx_pos,
                gaussians,
                scene,
                model_params,
                use_cuda_rasterizer,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

    return 1000.0 * elapsed / max(total, 1)


# ----------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------
SUMMARY_COLUMNS = [
    "variant",
    "letter",
    "label",
    "status",
    "reason",
    "model_path",
    "num_epochs",
    "target_gaussians",
    "num_gaussians",
    "render_k_rx",
    "render_k_tx",
    "rasterizer_backend",
    "num_evaluated",
    "nmse_mean_dB",
    "nmse_median_dB",
    "nmse_meanlinear_dB",
    "nmse_shape_mean_dB",
    "topk_acc_K1",
    "topk_acc_K4",
    "topk_acc_K8",
    "power_capture_K4",
    "anchor_dist_mean_norm",
    "anchor_dist_mean_m",
    "gain_position_rel_spread",
    "render_ms_per_map",
    "render_ms_per_map_reference",
    "train_seconds",
]


def write_summary_csv(path: str, rows: List[Dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_COLUMNS})


def save_figure(figure, output_dir: str, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, f"{stem}.png"), dpi=200)
    figure.savefig(os.path.join(output_dir, f"{stem}.pdf"))
    plt.close(figure)


def plot_variant_bars(output_dir: str, rows: List[Dict[str, object]]) -> bool:
    usable = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("nmse_mean_dB") not in ("", None)
    ]
    if not usable:
        return False

    labels = [str(row["label"]) for row in usable]
    nmse = [float(row["nmse_mean_dB"]) for row in usable]
    accuracy = [float(row["topk_acc_K1"]) for row in usable]
    positions = np.arange(len(usable))

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    axes[0].bar(positions, nmse, color="tab:blue", width=0.6)
    axes[0].set_ylabel("mean NMSE [dB]  (lower is better)")
    axes[0].set_title("Rendering NMSE per ablation variant")
    axes[0].axhline(nmse[0], color="0.25", linestyle="--", linewidth=1.2,
                    label="baseline")
    axes[0].legend(fontsize=8, loc="lower right")
    # NMSE is negative, so the bars hang below zero; give the axis headroom and
    # print each value just inside the end of its bar instead of off-canvas.
    axes[0].set_ylim(min(nmse) * 1.18, 0.0)
    for x, value in zip(positions, nmse):
        axes[0].annotate(f"{value:.2f}", (x, value), ha="center", va="bottom",
                         xytext=(0, 5), textcoords="offset points", fontsize=8,
                         color="white", fontweight="bold")

    axes[1].bar(positions, accuracy, color="tab:orange", width=0.6)
    axes[1].set_ylabel("mean top-1 beam-pair accuracy")
    axes[1].set_title("Dominant beam-pair accuracy per variant")
    axes[1].set_ylim(0.0, 1.0)
    for x, value in zip(positions, accuracy):
        axes[1].annotate(f"{value:.3f}", (x, value), ha="center", va="bottom",
                         xytext=(0, 3), textcoords="offset points", fontsize=8)

    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.3, linewidth=0.5)

    save_figure(figure, output_dir, "fig_ablation_bars")
    return True


def plot_topk_tradeoff(output_dir: str, sweep_rows: List[Dict[str, object]]) -> bool:
    usable = [row for row in sweep_rows if row.get("status") == "ok"]
    if not usable:
        return False

    labels = [f"({row['render_k_rx']},{row['render_k_tx']})" for row in usable]
    positions = np.arange(len(usable))
    nmse = [float(row["nmse_mean_dB"]) for row in usable]
    reference_ms = [float(row["render_ms_per_map_reference"]) for row in usable]
    deployed_ms = [
        float(row["render_ms_per_map"]) if row["render_ms_per_map"] != "" else np.nan
        for row in usable
    ]

    figure, axis = plt.subplots(figsize=(7.8, 4.8))

    nmse_line = axis.plot(
        positions, nmse, marker="o", color="tab:blue", linewidth=1.8,
        label="mean NMSE [dB]",
    )
    axis.set_xlabel("rendering top-K  (K_rx, K_tx)")
    axis.set_ylabel("mean NMSE [dB]", color="tab:blue")
    axis.tick_params(axis="y", labelcolor="tab:blue")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.grid(alpha=0.3, linewidth=0.5)

    time_axis = axis.twinx()
    time_line = time_axis.plot(
        positions, reference_ms, marker="s", color="tab:red", linewidth=1.8,
        label="render time [ms/map] (reference path)",
    )
    fused_line = time_axis.plot(
        positions, deployed_ms, marker="^", color="tab:green", linewidth=1.4,
        linestyle="--", label=f"render time [ms/map] (fused CUDA, K<={CUDA_MAX_TOPK})",
    )
    time_axis.set_ylabel("render time per map [ms]", color="tab:red")
    time_axis.tick_params(axis="y", labelcolor="tab:red")
    time_axis.set_yscale("log")

    handles = nmse_line + time_line + fused_line
    axis.legend(handles, [h.get_label() for h in handles], fontsize=8, loc="center left")
    axis.set_title("Rendering top-K: quality vs. cost (baseline checkpoint)")

    save_figure(figure, output_dir, "fig_topk_tradeoff")
    return True


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    default_epochs = int(repo_default(ModelParams, "num_epochs", 10))
    default_source = str(repo_default(ModelParams, "source_path", ""))
    default_gaussians = int(repo_default(ModelParams, "target_gaussians", 25_000))

    parser = argparse.ArgumentParser(description="MIMO-GS ablation study runner (E3)")
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--source_path", type=str, default=default_source)
    parser.add_argument("--target_gaussians", type=int, default=default_gaussians)
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Only re-evaluate the ablation checkpoints that already exist.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="",
        help="Comma-separated subset, by letter (a,b,c,d,e) or name "
        "(baseline,tied_anchor,tied_covariance,static_gain,topk_sweep).",
    )
    parser.add_argument("--outputs_root", type=str, default="outputs")
    parser.add_argument("--analysis_dir", type=str, default=os.path.join("analysis",
                                                                        "ablation"))
    parser.add_argument("--batch_size", type=int, default=0)
    return parser.parse_args()


def select_variants(selection: str) -> Tuple[List[Dict[str, object]], bool]:
    """Resolve --variants into training variants plus the sweep flag."""
    if not selection.strip():
        return list(TRAINING_VARIANTS), True

    requested = {token.strip().lower() for token in selection.split(",") if token.strip()}
    known = {variant["letter"]: variant for variant in TRAINING_VARIANTS}
    known.update({variant["key"]: variant for variant in TRAINING_VARIANTS})

    chosen: List[Dict[str, object]] = []
    for variant in TRAINING_VARIANTS:
        if variant["letter"] in requested or variant["key"] in requested:
            chosen.append(variant)

    run_sweep = SWEEP_LETTER in requested or SWEEP_KEY in requested

    unknown = requested - set(known) - {SWEEP_LETTER, SWEEP_KEY}
    if unknown:
        raise SystemExit(f"[run_ablation] Unknown variant(s): {sorted(unknown)}")

    return chosen, run_sweep


def main() -> None:
    arguments = parse_arguments()
    repository_root = os.path.dirname(os.path.abspath(__file__))

    source_path = arguments.source_path
    if not os.path.isabs(source_path):
        source_path = os.path.join(repository_root, source_path)
    source_path = os.path.abspath(source_path)
    if not os.path.isdir(source_path):
        raise SystemExit(f"[run_ablation] Dataset directory '{source_path}' is missing.")

    outputs_root = os.path.join(repository_root, arguments.outputs_root)
    analysis_dir = os.path.join(repository_root, arguments.analysis_dir)
    os.makedirs(outputs_root, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    variants, run_sweep = select_variants(arguments.variants)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_lines: List[str] = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    log("=" * 78)
    log("[run_ablation] MIMO-GS ablation study (E3)")
    log("=" * 78)
    log(f"  device            : {device}")
    log(f"  source_path       : {source_path}")
    log(f"  epochs            : {arguments.epochs}")
    log(f"  target_gaussians  : {arguments.target_gaussians}")
    log(f"  training variants : {[v['key'] for v in variants] or 'none'}")
    log(f"  eval-time sweep   : {'yes' if run_sweep else 'no'}")
    log(f"  skip_training     : {arguments.skip_training}")
    log("")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    train_state: Dict[str, Dict[str, object]] = {}

    for variant in variants:
        key = str(variant["key"])
        model_path = os.path.join(outputs_root, f"ablation_{key}")
        train_log = os.path.join(analysis_dir, f"train_{key}.log")
        command = build_training_command(
            variant, model_path, arguments.epochs, source_path,
            arguments.target_gaussians,
        )

        log("-" * 78)
        log(f"[run_ablation] VARIANT {variant['letter'].upper()} :: {variant['label']}")
        log(f"  {variant['description']}")
        log(f"  model_path : {model_path}")
        log(f"  command    : {' '.join(command)}")

        if arguments.skip_training:
            exists = os.path.isfile(os.path.join(model_path, eval_render.CHECKPOINT_NAME))
            log(f"  training   : SKIPPED (--skip_training), checkpoint present={exists}")
            train_state[key] = {
                "model_path": model_path,
                "ok": exists,
                "seconds": float("nan"),
                "reason": "" if exists else "no existing checkpoint to re-evaluate",
            }
            continue

        success, seconds, reason = run_training(command, repository_root, train_log)
        if success:
            log(f"  training   : OK in {seconds / 60.0:.1f} min (log: {train_log})")
        else:
            log(f"  training   : FAILED -- {reason}")

        train_state[key] = {
            "model_path": model_path,
            "ok": success,
            "seconds": seconds,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    log("")
    log("=" * 78)
    log("[run_ablation] Evaluating variants on the shared test set")
    log("=" * 78)

    rows: List[Dict[str, object]] = []
    baseline_run_dir: Optional[str] = None

    for variant in variants:
        key = str(variant["key"])
        state = train_state[key]
        base_row: Dict[str, object] = {
            "variant": key,
            "letter": variant["letter"],
            "label": variant["label"],
            "model_path": os.path.relpath(str(state["model_path"]), repository_root),
            "num_epochs": arguments.epochs,
            "target_gaussians": arguments.target_gaussians,
            "train_seconds": (
                "" if math.isnan(float(state["seconds"]))
                else f"{float(state['seconds']):.1f}"
            ),
        }

        if not state["ok"]:
            base_row["status"] = "failed"
            base_row["reason"] = str(state["reason"]) or "training did not complete"
            rows.append(base_row)
            log(f"[{key}] skipped evaluation -- {base_row['reason']}")
            continue

        try:
            scene, gaussians, model_params, _, checkpoint = load_trained_run(
                str(state["model_path"]), device
            )
        except Exception as error:  # noqa: BLE001 - keep the study going
            base_row["status"] = "failed"
            base_row["reason"] = f"could not load checkpoint: {error}"
            rows.append(base_row)
            log(f"[{key}] {base_row['reason']}")
            continue

        batch_size = int(arguments.batch_size) or int(
            getattr(model_params, "batch_size", 8)
        )
        k_rx = int(getattr(model_params, "max_active_rx_beams", 4))
        k_tx = int(getattr(model_params, "max_active_tx_beams", 4))
        use_cuda, backend = backend_for(k_rx, k_tx, device.type == "cuda")

        results = eval_render.evaluate_test_set(
            scene, gaussians, model_params, device, batch_size, use_cuda
        )

        scale_factor = float(getattr(scene.test_set, "scale_factor", 1.0))
        anchor_distance = mean_anchor_distance(gaussians)
        gain_spread = verify_static_gain(gaussians, scene, device)

        base_row.update(collect_metrics(results))
        base_row.update(
            {
                "status": "ok",
                "reason": "",
                "num_gaussians": int(gaussians.get_xyz.shape[0]),
                "render_k_rx": k_rx,
                "render_k_tx": k_tx,
                "rasterizer_backend": backend,
                "anchor_dist_mean_norm": f"{anchor_distance:.8f}",
                "anchor_dist_mean_m": f"{anchor_distance * scale_factor:.6f}",
                "gain_position_rel_spread": f"{gain_spread:.8f}",
            }
        )
        rows.append(base_row)

        log(
            f"[{key}] NMSE mean {base_row['nmse_mean_dB']:.2f} dB | "
            f"median {base_row['nmse_median_dB']:.2f} dB | "
            f"top-1 {base_row['topk_acc_K1']:.4f} | "
            f"anchor dist {anchor_distance * scale_factor:.4f} m | "
            f"gain spread {gain_spread:.3e} | "
            f"N={base_row['num_gaussians']}"
        )

        if key == "baseline":
            baseline_run_dir = str(state["model_path"])

        del scene, gaussians, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # (e) eval-time rendering top-K sweep on the baseline checkpoint
    # ------------------------------------------------------------------
    sweep_rows: List[Dict[str, object]] = []

    if run_sweep:
        log("")
        log("-" * 78)
        log(f"[run_ablation] VARIANT E :: (e) rendering top-K sweep")

        if baseline_run_dir is None:
            candidate = os.path.join(outputs_root, "ablation_baseline")
            if os.path.isfile(os.path.join(candidate, eval_render.CHECKPOINT_NAME)):
                baseline_run_dir = candidate

        if baseline_run_dir is None:
            reason = "no baseline checkpoint available for the eval-time sweep"
            log(f"  SKIPPED -- {reason}")
            sweep_rows.append(
                {
                    "variant": SWEEP_KEY,
                    "letter": SWEEP_LETTER,
                    "label": "(e) top-K sweep",
                    "status": "failed",
                    "reason": reason,
                    "num_epochs": arguments.epochs,
                }
            )
        else:
            scene, gaussians, model_params, _, _ = load_trained_run(
                baseline_run_dir, device
            )
            batch_size = int(arguments.batch_size) or int(
                getattr(model_params, "batch_size", 8)
            )
            num_rx_beams = int(scene.beam_rows)
            num_tx_beams = int(scene.beam_cols)

            settings = [(1, 1), (2, 2), (4, 4), (8, 8), (num_rx_beams, num_tx_beams)]
            # Drop duplicates while keeping order, in case (8,8) already is the
            # untruncated setting for a small array.
            settings = [
                (min(kr, num_rx_beams), min(kt, num_tx_beams)) for kr, kt in settings
            ]
            seen = set()
            unique_settings = []
            for setting in settings:
                if setting not in seen:
                    seen.add(setting)
                    unique_settings.append(setting)

            log(f"  baseline checkpoint : {baseline_run_dir}")
            log(f"  settings            : {unique_settings} "
                f"(full grid = ({num_rx_beams},{num_tx_beams}))")

            for k_rx, k_tx in unique_settings:
                overridden = params_with_beam_limits(model_params, k_rx, k_tx)
                use_cuda, backend = backend_for(k_rx, k_tx, device.type == "cuda")

                results = eval_render.evaluate_test_set(
                    scene, gaussians, overridden, device, batch_size, use_cuda
                )

                # The reference path covers every K, so it gives one internally
                # consistent timing curve; the fused kernel is timed separately
                # where it is usable (K <= CUDA_MAX_TOPK).
                reference_ms = measure_render_time(
                    scene, gaussians, overridden, device, batch_size, False
                )
                fused_ms = (
                    measure_render_time(
                        scene, gaussians, overridden, device, batch_size, True
                    )
                    if use_cuda
                    else ""
                )

                row: Dict[str, object] = {
                    "variant": f"{SWEEP_KEY}_K{k_rx}x{k_tx}",
                    "letter": SWEEP_LETTER,
                    "label": f"(e) K=({k_rx},{k_tx})",
                    "status": "ok",
                    "reason": "",
                    "model_path": os.path.relpath(baseline_run_dir, repository_root),
                    "num_epochs": arguments.epochs,
                    "target_gaussians": arguments.target_gaussians,
                    "num_gaussians": int(gaussians.get_xyz.shape[0]),
                    "render_k_rx": k_rx,
                    "render_k_tx": k_tx,
                    "rasterizer_backend": backend,
                    "render_ms_per_map": (
                        "" if fused_ms == "" else f"{float(fused_ms):.4f}"
                    ),
                    "render_ms_per_map_reference": f"{reference_ms:.4f}",
                }
                row.update(collect_metrics(results))
                sweep_rows.append(row)

                log(
                    f"  K=({k_rx},{k_tx}) [{backend}] NMSE {row['nmse_mean_dB']:.2f} dB | "
                    f"top-1 {row['topk_acc_K1']:.4f} | "
                    f"reference {reference_ms:.3f} ms/map"
                    + ("" if fused_ms == "" else f" | fused {float(fused_ms):.3f} ms/map")
                )

            del scene, gaussians
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    all_rows = rows + sweep_rows
    summary_path = os.path.join(analysis_dir, "ablation_summary.csv")
    write_summary_csv(summary_path, all_rows)

    bars_written = plot_variant_bars(analysis_dir, rows)
    tradeoff_written = plot_topk_tradeoff(analysis_dir, sweep_rows)

    log("")
    log("=" * 78)
    log("[run_ablation] SUMMARY")
    log("=" * 78)
    header = (
        f"  {'variant':<22}{'status':<9}{'NMSE mean':>11}{'NMSE med':>10}"
        f"{'top-1':>9}{'top-4':>9}{'cap@4':>9}"
    )
    log(header)
    for row in all_rows:
        if row.get("status") != "ok":
            log(f"  {str(row['variant']):<22}{'FAILED':<9}  {row.get('reason', '')}")
            continue
        log(
            f"  {str(row['variant']):<22}{'ok':<9}"
            f"{float(row['nmse_mean_dB']):>11.2f}"
            f"{float(row['nmse_median_dB']):>10.2f}"
            f"{float(row['topk_acc_K1']):>9.4f}"
            f"{float(row['topk_acc_K4']):>9.4f}"
            f"{float(row['power_capture_K4']):>9.4f}"
        )

    log("")
    log(f"  summary csv        : {summary_path}")
    log(f"  fig_ablation_bars  : {'written' if bars_written else 'skipped (no data)'}")
    log(f"  fig_topk_tradeoff  : {'written' if tradeoff_written else 'skipped (no data)'}")
    log("=" * 78)

    with open(os.path.join(analysis_dir, "ablation_log.txt"), "w",
              encoding="utf-8") as handle:
        handle.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
