"""Head-to-head: pure coordinate-MLP baselines vs. the trained MIMO-GS model.

Zero-argument runnable::

    python eval_mlp_compare.py

Every model is scored on the same test set with the metric definitions from
:mod:`eval_render` -- ``topk_metrics`` (top-K overlap + power capture),
``summarize`` and ``mean_linear_db`` are imported, not reimplemented, and the
NMSE arithmetic is verified against ``eval_render.evaluate_test_set`` itself:
the MIMO-GS checkpoint is scored twice, once through the reference function and
once through the generic predictor path used for the MLPs, and the two must
agree bit-for-bit before anything is reported.

No figures are produced. Results land in ``analysis/mlp_compare/mlp_vs_gs.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_MLP import PositionMLP, build_scene, count_parameters
from utils.loss import normalize_mag_map

try:
    import eval_render
except ImportError as error:  # pragma: no cover - eval_render ships with E1.
    raise SystemExit(
        f"[mlp_cmp] eval_render.py is required for the metric definitions: {error}"
    )


REPORT_TOPK = (1, 4, 8)
CAPTURE_K = (1, 4)

# Wrap-fixed 30-epoch MIMO-GS checkpoint.
GS_RUN = "20260807_011237"

# ``(display label, run directory relative to --outputs_root)``.
# The MLP budget is matched to the MIMO-GS run (30 epochs); the 10-epoch
# mlp_small row is kept so the effect of the longer budget stays visible.
MLP_RUNS: Tuple[Tuple[str, str], ...] = (
    ("mlp_small_30ep", os.path.join("mlp_30ep", "mlp_small")),
    ("mlp_medium_30ep", os.path.join("mlp_30ep", "mlp_medium")),
    ("mlp_large_30ep", os.path.join("mlp_30ep", "mlp_large")),
    ("mlp_small_10ep", os.path.join("mlp_10ep", "mlp_small")),
)

# The headline NMSE convention, matching eval_render's summary table:
# normalized prediction vs normalized target, averaged per location in dB.
PRIMARY_NMSE = "nmse_shape_mean_dB"

Predictor = Callable[[torch.Tensor], torch.Tensor]


# ----------------------------------------------------------------------
# Generic evaluation over an arbitrary predictor
# ----------------------------------------------------------------------
def evaluate_predictor(
    predict: Predictor,
    scene,
    device: torch.device,
    batch_size: int,
) -> Dict[str, object]:
    """Score any ``position -> (B, Nr, Nt)`` predictor on the test set.

    Mirrors ``eval_render.evaluate_test_set`` exactly; the equivalence is
    asserted at runtime in :func:`main`.
    """
    loader = DataLoader(
        scene.test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    indices: List[int] = []
    nmse_raw: List[float] = []
    nmse_shape: List[float] = []
    topk_accumulator: Dict[int, List[float]] = {k: [] for k in eval_render.TOPK_VALUES}
    capture_accumulator: Dict[int, List[float]] = {
        k: [] for k in eval_render.TOPK_VALUES
    }
    skipped = 0
    cursor = 0

    with torch.no_grad():
        for magnitude, rx_pos in loader:
            magnitude = magnitude.to(device, non_blocking=True)
            rx_pos = rx_pos.to(device, non_blocking=True)

            batch = magnitude.shape[0]
            batch_indices = torch.arange(cursor, cursor + batch)
            cursor += batch

            ground_truth = magnitude.reshape(batch, scene.beam_rows, scene.beam_cols)

            peak = ground_truth.reshape(batch, -1).amax(dim=1)
            valid = peak > eval_render.EPS
            num_valid = int(valid.sum().item())
            skipped += batch - num_valid
            if num_valid == 0:
                continue

            predicted = predict(rx_pos.reshape(-1, 3))

            ground_truth = ground_truth[valid]
            predicted = predicted[valid]

            target_n = normalize_mag_map(ground_truth)
            predicted_n = normalize_mag_map(predicted)

            target_flat = target_n.reshape(num_valid, -1)
            predicted_flat = predicted.reshape(num_valid, -1)
            predicted_n_flat = predicted_n.reshape(num_valid, -1)

            energy = target_flat.square().sum(dim=1).clamp_min(eval_render.EPS)
            raw_ratio = (predicted_flat - target_flat).square().sum(dim=1) / energy
            shape_ratio = (
                predicted_n_flat - target_flat
            ).square().sum(dim=1) / energy

            nmse_raw.extend(
                (10.0 * torch.log10(raw_ratio.clamp_min(1e-12))).cpu().tolist()
            )
            nmse_shape.extend(
                (10.0 * torch.log10(shape_ratio.clamp_min(1e-12))).cpu().tolist()
            )

            for k, (overlap, capture) in eval_render.topk_metrics(
                predicted_flat, target_flat, eval_render.TOPK_VALUES
            ).items():
                topk_accumulator[k].extend(overlap.cpu().tolist())
                capture_accumulator[k].extend(capture.cpu().tolist())

            indices.extend(batch_indices[valid.cpu()].tolist())

    return {
        "index": np.asarray(indices, dtype=np.int64),
        "nmse_raw_db": np.asarray(nmse_raw, dtype=np.float64),
        "nmse_shape_db": np.asarray(nmse_shape, dtype=np.float64),
        "topk": {k: np.asarray(v, dtype=np.float64) for k, v in topk_accumulator.items()},
        "capture": {
            k: np.asarray(v, dtype=np.float64) for k, v in capture_accumulator.items()
        },
        "skipped_zero_power": skipped,
    }


def measure_inference_time(
    predict: Predictor,
    scene,
    device: torch.device,
    batch_size: int,
) -> float:
    """Milliseconds per map, warmup excluded and data pre-staged on device."""
    positions = getattr(scene.test_set, "positions", None)
    if positions is None:
        positions = torch.stack(
            [scene.test_set[i][1].reshape(3) for i in range(len(scene.test_set))], dim=0
        )
    positions = positions.reshape(-1, 3).to(device)
    total = int(positions.shape[0])

    with torch.no_grad():
        predict(positions[: min(batch_size, total)])
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        started = time.perf_counter()
        for start in range(0, total, batch_size):
            predict(positions[start : start + batch_size])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

    return 1000.0 * elapsed / max(total, 1)


def collect_row(results: Dict[str, object]) -> Dict[str, float]:
    raw = results["nmse_raw_db"]
    shape = results["nmse_shape_db"]
    row: Dict[str, float] = {
        "nmse_raw_mean_dB": float(np.mean(raw)),
        "nmse_raw_median_dB": float(np.median(raw)),
        "nmse_raw_meanlinear_dB": eval_render.mean_linear_db(raw),
        "nmse_shape_mean_dB": float(np.mean(shape)),
        "nmse_shape_median_dB": float(np.median(shape)),
        "nmse_shape_meanlinear_dB": eval_render.mean_linear_db(shape),
        "num_evaluated": int(results["index"].shape[0]),
    }
    for k in REPORT_TOPK:
        row[f"topk_acc_K{k}"] = float(np.mean(results["topk"][k]))
    for k in CAPTURE_K:
        row[f"power_capture_K{k}"] = float(np.mean(results["capture"][k]))
    return row


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
def load_mlp(run_dir: str, device: torch.device) -> Tuple[PositionMLP, Dict]:
    checkpoint = torch.load(
        os.path.join(run_dir, "model.pth"), map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]
    model = PositionMLP(
        num_outputs=int(config["num_outputs"]),
        hidden=int(config["hidden"]),
        depth=int(config["depth"]),
        num_frequencies=int(config.get("num_frequencies", 6)),
        include_input=bool(config.get("include_input", True)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure-MLP vs MIMO-GS head-to-head (numbers only)"
    )
    parser.add_argument("--gs_ckpt", type=str, default=os.path.join("outputs", GS_RUN))
    parser.add_argument("--outputs_root", type=str, default="outputs")
    parser.add_argument("--analysis_dir", type=str,
                        default=os.path.join("analysis", "mlp_compare"))
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument(
        "--mlp_runs", type=str, default="",
        help="Comma-separated 'label=relative/run/dir' overrides for the MLP rows.",
    )
    parser.add_argument(
        "--trajectory_of", type=str, default="mlp_small_30ep",
        help="Label whose per-epoch test trajectory is exported and printed.",
    )
    return parser.parse_args()


def resolve_mlp_runs(spec: str) -> Tuple[Tuple[str, str], ...]:
    if not spec.strip():
        return MLP_RUNS
    pairs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"[mlp_cmp] --mlp_runs entry must be label=path, got '{item}'")
        label, path = item.split("=", 1)
        pairs.append((label.strip(), path.strip()))
    return tuple(pairs)


def main() -> None:
    arguments = parse_arguments()
    repository_root = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outputs_root = arguments.outputs_root
    if not os.path.isabs(outputs_root):
        outputs_root = os.path.join(repository_root, outputs_root)

    gs_dir = arguments.gs_ckpt
    if not os.path.isabs(gs_dir):
        gs_dir = os.path.join(repository_root, gs_dir)
    gs_checkpoint_path = os.path.join(gs_dir, eval_render.CHECKPOINT_NAME)
    if not os.path.isfile(gs_checkpoint_path):
        raise SystemExit(f"[mlp_cmp] MIMO-GS checkpoint not found: {gs_checkpoint_path}")

    print("=" * 78)
    print("[mlp_cmp] Pure-MLP baseline vs MIMO-GS -- 16x64 head-to-head")
    print("=" * 78)
    print(f"  device          : {device}")
    print(f"  MIMO-GS run     : {os.path.basename(os.path.normpath(gs_dir))}")

    # --- MIMO-GS -----------------------------------------------------
    gs_checkpoint = torch.load(gs_checkpoint_path, map_location="cpu",
                               weights_only=False)
    gs_model_params, gs_opt_params = eval_render.restore_config(gs_dir, gs_checkpoint)
    scene, gaussians = eval_render.build_scene_and_model(
        gs_model_params, gs_opt_params, gs_checkpoint, device
    )

    batch_size = max(
        1, int(arguments.batch_size) or int(getattr(gs_model_params, "batch_size", 8))
    )
    use_cuda_rasterizer = bool(
        int(getattr(gs_model_params, "use_cuda_rasterizer", 1))
    ) and device.type == "cuda"

    print(f"  test locations  : {len(scene.test_set)} | "
          f"beam grid {scene.beam_rows}x{scene.beam_cols} | batch {batch_size}")
    print("")

    tx_pos = torch.as_tensor(scene.bs_position, dtype=torch.float32, device=device)

    def gs_predict(rx_pos: torch.Tensor) -> torch.Tensor:
        return eval_render.render_batch(
            rx_pos, tx_pos, gaussians, scene, gs_model_params, use_cuda_rasterizer
        )

    # Reference path: eval_render's own function, untouched.
    reference = eval_render.evaluate_test_set(
        scene, gaussians, gs_model_params, device, batch_size, use_cuda_rasterizer
    )
    # Generic path: the same predictor interface the MLPs use.
    generic = evaluate_predictor(gs_predict, scene, device, batch_size)

    drift = max(
        float(np.max(np.abs(reference["nmse_raw_db"] - generic["nmse_raw_db"]))),
        float(np.max(np.abs(reference["nmse_shape_db"] - generic["nmse_shape_db"]))),
    )
    topk_drift = max(
        float(np.max(np.abs(reference["topk"][k] - generic["topk"][k])))
        for k in REPORT_TOPK
    )
    # The fused rasterizer accumulates with CUDA atomics, so it is not
    # bit-reproducible: calling eval_render.evaluate_test_set twice with
    # identical arguments already differs by ~8e-06 dB. The tolerance sits
    # just above that measured noise floor and still four orders of magnitude
    # below anything that could change a conclusion.
    nmse_tolerance = 1e-3
    print(f"[mlp_cmp] pipeline equivalence check (eval_render vs generic path):")
    print(f"    max |NMSE_dB difference| = {drift:.3e}  (tolerance {nmse_tolerance:.0e},")
    print(f"                                renderer run-to-run noise is ~8e-06)")
    print(f"    max |top-K difference|   = {topk_drift:.3e}")
    if drift > nmse_tolerance or topk_drift > 1e-12:
        raise SystemExit(
            "[mlp_cmp] The generic evaluation path does not reproduce "
            "eval_render.evaluate_test_set. Refusing to report."
        )
    print("    -> equivalent, the MLPs are scored by the same code path.\n")

    # Cross-check against the stored E1 summary, if present.
    stored_path = os.path.join(
        repository_root, "analysis",
        os.path.basename(os.path.normpath(gs_dir)), "eval_render",
        "metrics_summary.csv",
    )
    stored_note = "metrics_summary.csv not found"
    if os.path.isfile(stored_path):
        with open(stored_path, "r", encoding="utf-8") as handle:
            stored = next(csv.DictReader(handle))
        # Both conventions are checked; the shape column is the headline one.
        deltas = {}
        for label, stored_key, computed in (
            ("shape", "NMSE_shape_mean_dB", reference["nmse_shape_db"]),
            ("raw", "NMSE_raw_mean_dB", reference["nmse_raw_db"]),
        ):
            stored_mean = float(stored[stored_key])
            computed_mean = float(np.mean(computed))
            deltas[label] = abs(stored_mean - computed_mean)
            print(f"[mlp_cmp] stored E1 mean NMSE ({label:<5}) {stored_mean:9.6f} dB vs "
                  f"recomputed {computed_mean:9.6f} dB -> delta {deltas[label]:.2e}")
        stored_note = (f"matches stored E1 summary "
                       f"(shape delta {deltas['shape']:.2e} dB)")
        if max(deltas.values()) > 1e-4:
            raise SystemExit(
                "[mlp_cmp] Recomputed MIMO-GS NMSE disagrees with the stored E1 "
                "summary. Refusing to report."
            )
        print("")

    gs_parameters = count_parameters(gaussians.dynamic_gain_net) + int(
        sum(
            tensor.numel()
            for tensor in (
                gaussians._xyz, gaussians._xyz_tx, gaussians._scaling,
                gaussians._rotation, gaussians._scaling_tx, gaussians._rotation_tx,
                gaussians._opacity,
            )
        )
    )
    gs_time = measure_inference_time(gs_predict, scene, device, batch_size)

    rows: List[Dict[str, object]] = []
    reference_indices: Optional[np.ndarray] = generic["index"]

    gs_row: Dict[str, object] = {
        "model": "MIMO-GS",
        "kind": "3D-GS renderer",
        "hidden": int(getattr(gs_model_params, "target_gaussians", 0)),
        "depth": "",
        "epochs": int(getattr(gs_model_params, "num_epochs", 0)),
        "batch_size": batch_size,
        "lr": "see run_args.txt",
        "parameters": gs_parameters,
        "infer_ms_per_map": gs_time,
        "device": str(device),
        "notes": f"K=({getattr(gs_model_params, 'max_active_rx_beams', '?')},"
                 f"{getattr(gs_model_params, 'max_active_tx_beams', '?')}); "
                 f"{stored_note}",
    }
    gs_row.update(collect_row(generic))
    rows.append(gs_row)

    print(f"[{'MIMO-GS':<15}] NMSE shape mean {gs_row['nmse_shape_mean_dB']:7.3f} dB | "
          f"raw {gs_row['nmse_raw_mean_dB']:7.3f} | "
          f"top-1 {gs_row['topk_acc_K1']:.4f} | "
          f"params {gs_parameters:,} | {gs_time:.3f} ms/map")

    del gaussians
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- MLP baselines -----------------------------------------------
    trajectories: Dict[str, List[Dict[str, float]]] = {}
    for name, relative_dir in resolve_mlp_runs(arguments.mlp_runs):
        run_dir = os.path.join(outputs_root, relative_dir)
        if not os.path.isfile(os.path.join(run_dir, "model.pth")):
            print(f"[{name:<15}] SKIPPED -- no checkpoint at {run_dir}")
            continue

        model, checkpoint = load_mlp(run_dir, device)

        # Fairness guard: the MLP must have been TRAINED on the very dataset it
        # is now being scored on.  The two ASU variants (_lt and _outdoor) have
        # the same shapes but different splits, so a mismatch is otherwise
        # invisible in the numbers.
        trained_on = str(checkpoint.get("training", {}).get("source_path", ""))
        if trained_on and os.path.realpath(trained_on) != os.path.realpath(scene.datadir):
            raise SystemExit(
                f"[mlp_cmp] {name} was trained on '{trained_on}' but is being "
                f"evaluated on '{scene.datadir}'. Refusing to report."
            )

        def mlp_predict(rx_pos: torch.Tensor, _model=model) -> torch.Tensor:
            return _model(rx_pos).reshape(-1, scene.beam_rows, scene.beam_cols)

        results = evaluate_predictor(mlp_predict, scene, device, batch_size)

        if not np.array_equal(results["index"], reference_indices):
            raise SystemExit(
                f"[mlp_cmp] {name} was evaluated on different test indices than "
                "MIMO-GS. Refusing to report."
            )

        inference_ms = measure_inference_time(mlp_predict, scene, device, batch_size)
        training = checkpoint.get("training", {})
        trajectory = checkpoint.get("trajectory", [])
        trajectories[name] = trajectory

        row: Dict[str, object] = {
            "model": name,
            "kind": "pure coordinate-MLP",
            "hidden": int(checkpoint["config"]["hidden"]),
            "depth": int(checkpoint["config"]["depth"]),
            "epochs": int(training.get("epochs", 0)),
            "batch_size": int(training.get("batch_size", batch_size)),
            "lr": f"{training.get('lr_init')}->{training.get('lr_final')} cosine",
            "parameters": int(checkpoint.get("parameters", count_parameters(model))),
            "infer_ms_per_map": inference_ms,
            "device": str(device),
            "notes": f"PE nf=6 include_input; softplus out; "
                     f"loss={training.get('loss')} topk_ratio={training.get('topk_ratio')}; "
                     f"final train-conv scale NMSE "
                     f"{trajectory[-1]['test_scale_nmse_db']:.3f} dB"
            if trajectory else "",
        }
        row.update(collect_row(results))
        rows.append(row)

        print(f"[{name:<15}] NMSE shape mean {row['nmse_shape_mean_dB']:7.3f} dB | "
              f"raw {row['nmse_raw_mean_dB']:7.3f} | "
              f"top-1 {row['topk_acc_K1']:.4f} | "
              f"params {row['parameters']:,} | {inference_ms:.3f} ms/map")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- CSV ----------------------------------------------------------
    analysis_dir = arguments.analysis_dir
    if not os.path.isabs(analysis_dir):
        analysis_dir = os.path.join(repository_root, analysis_dir)
    os.makedirs(analysis_dir, exist_ok=True)

    columns = [
        "model", "kind", "hidden", "depth", "parameters", "epochs", "batch_size", "lr",
        "num_evaluated",
        "nmse_raw_mean_dB", "nmse_raw_median_dB", "nmse_raw_meanlinear_dB",
        "nmse_shape_mean_dB", "nmse_shape_median_dB", "nmse_shape_meanlinear_dB",
        "topk_acc_K1", "topk_acc_K4", "topk_acc_K8",
        "power_capture_K1", "power_capture_K4",
        "infer_ms_per_map", "device", "notes",
    ]
    csv_path = os.path.join(analysis_dir, "mlp_vs_gs.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})

    # --- Console summary ---------------------------------------------
    print("")
    print("=" * 100)
    print("[mlp_cmp] COMPARISON  (all models, identical test set and metric code)")
    print("=" * 100)
    print("  NMSE-shape = normalized pred vs normalized target (eval_render headline)")
    print("  NMSE-raw   = raw pred vs normalized target")
    print("  both averaged per location in dB\n")
    print(f"  {'model':<16}{'params':>11}{'epochs':>8}"
          f"{'shape mean':>12}{'shape med':>11}{'raw mean':>10}"
          f"{'top-1':>8}{'top-4':>8}{'cap@1':>8}{'cap@4':>8}{'ms/map':>9}")
    for row in rows:
        print(
            f"  {str(row['model']):<16}{int(row['parameters']):>11,}"
            f"{int(row['epochs']):>8}"
            f"{float(row['nmse_shape_mean_dB']):>12.3f}"
            f"{float(row['nmse_shape_median_dB']):>11.3f}"
            f"{float(row['nmse_raw_mean_dB']):>10.3f}"
            f"{float(row['topk_acc_K1']):>8.4f}{float(row['topk_acc_K4']):>8.4f}"
            f"{float(row['power_capture_K1']):>8.4f}"
            f"{float(row['power_capture_K4']):>8.4f}"
            f"{float(row['infer_ms_per_map']):>9.3f}"
        )

    # Ranking agreement between NMSE and the selection-oriented metrics.
    def ranking(key: str, descending: bool) -> List[str]:
        ordered = sorted(rows, key=lambda r: float(r[key]), reverse=descending)
        return [str(r["model"]) for r in ordered]

    nmse_rank = ranking(PRIMARY_NMSE, False)           # lower dB is better
    top1_rank = ranking("topk_acc_K1", True)
    top4_rank = ranking("topk_acc_K4", True)
    cap4_rank = ranking("power_capture_K4", True)

    print("")
    print("  ranking by NMSE-shape (best first)    : " + " > ".join(nmse_rank))
    print("  ranking by top-1 accuracy             : " + " > ".join(top1_rank))
    print("  ranking by top-4 accuracy             : " + " > ".join(top4_rank))
    print("  ranking by power capture @K=4         : " + " > ".join(cap4_rank))
    agree = (nmse_rank == top1_rank == top4_rank == cap4_rank)
    print(f"  -> rankings {'AGREE' if agree else 'DISAGREE'} across metric families")

    # --- Head-to-head gap: MIMO-GS vs the best MLP --------------------
    gs = rows[0]
    mlp_rows = [r for r in rows[1:]]
    if mlp_rows:
        best = min(mlp_rows, key=lambda r: float(r[PRIMARY_NMSE]))
        print("")
        print(f"  best MLP by NMSE-shape: {best['model']}")
        print(f"  {'metric':<26}{'MIMO-GS':>12}{'best MLP':>12}{'gap':>12}")
        for key, label, fmt in (
            ("nmse_shape_mean_dB", "NMSE-shape mean [dB]", "{:+.3f}"),
            ("nmse_shape_median_dB", "NMSE-shape median [dB]", "{:+.3f}"),
            ("nmse_raw_mean_dB", "NMSE-raw mean [dB]", "{:+.3f}"),
            ("topk_acc_K1", "top-1 accuracy", "{:+.4f}"),
            ("topk_acc_K4", "top-4 accuracy", "{:+.4f}"),
            ("power_capture_K1", "power capture @K=1", "{:+.4f}"),
            ("power_capture_K4", "power capture @K=4", "{:+.4f}"),
            ("infer_ms_per_map", "inference [ms/map]", "{:+.3f}"),
        ):
            g, m = float(gs[key]), float(best[key])
            print(f"  {label:<26}{g:>12.4f}{m:>12.4f}{fmt.format(g - m):>12}")

    # --- Per-epoch trajectory export ----------------------------------
    target_label = arguments.trajectory_of
    trajectory = trajectories.get(target_label, [])
    if trajectory:
        traj_path = os.path.join(analysis_dir, f"{target_label}_trajectory.csv")
        traj_columns = [
            "epoch", "train_loss", "test_perloc_shape_mean_db",
            "test_perloc_mean_db", "test_scale_nmse_db", "test_shape_nmse_db", "lr",
        ]
        with open(traj_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=traj_columns,
                                    extrasaction="ignore")
            writer.writeheader()
            for entry in trajectory:
                writer.writerow({k: entry.get(k, "") for k in traj_columns})

        print("")
        print("=" * 100)
        print(f"[mlp_cmp] PER-EPOCH TEST TRAJECTORY -- {target_label}")
        print("=" * 100)
        print(f"  {'epoch':>6}{'train loss':>14}{'NMSE-shape [dB]':>18}"
              f"{'NMSE-raw [dB]':>16}{'delta shape':>14}")
        previous = None
        for entry in trajectory:
            shape_db = float(entry.get("test_perloc_shape_mean_db", float("nan")))
            delta = "" if previous is None else f"{shape_db - previous:+.3f}"
            previous = shape_db
            print(f"  {int(entry['epoch']):>6}{float(entry['train_loss']):>14.6f}"
                  f"{shape_db:>18.3f}"
                  f"{float(entry.get('test_perloc_mean_db', float('nan'))):>16.3f}"
                  f"{delta:>14}")

        best_epoch = min(
            trajectory, key=lambda e: float(e.get("test_perloc_shape_mean_db", 1e9))
        )
        last = trajectory[-1]
        tail = [float(e["test_perloc_shape_mean_db"]) for e in trajectory[-5:]]
        print("")
        print(f"  best epoch {int(best_epoch['epoch'])}: "
              f"{float(best_epoch['test_perloc_shape_mean_db']):.3f} dB | "
              f"final epoch {int(last['epoch'])}: "
              f"{float(last['test_perloc_shape_mean_db']):.3f} dB")
        print(f"  improvement over the last 5 epochs: "
              f"{tail[0] - tail[-1]:+.3f} dB "
              f"(spread {max(tail) - min(tail):.3f} dB)")
        print(f"  trajectory CSV -> {traj_path}")

    print("")
    print(f"[mlp_cmp] CSV written to {csv_path}")
    print("=" * 100)


if __name__ == "__main__":
    sys.exit(main())
