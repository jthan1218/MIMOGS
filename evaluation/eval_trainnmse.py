"""TRAIN-set NMSE for a single MIMO-GS checkpoint, next to its TEST-set NMSE.

Runs with zero arguments::

    python eval_trainnmse.py

Every other eval script in this repository reports the TEST split, which is a
*generalization* number.  It cannot answer the prior question: does the model
even fit the data it was optimized on?  This script scores the *same*
checkpoint on both splits with the *same* metric code, so the two numbers are
directly comparable and their difference is meaningful.

Nothing here is re-implemented.  Checkpoint restoration, scene/model
construction, rendering and the metric arithmetic are all imported from
``eval_render`` / ``eval_baseline_rt``; this file only chooses *which rows* to
feed them and prints the comparison.

Reading the result
------------------
* ``train ~= test``  -- the model is *fitting-capacity* limited.  It cannot even
  reproduce its own training locations, so more data or stronger regularization
  will not help; the representation itself is the bottleneck.
* ``train << test``  -- a genuine *generalization gap*.  The model memorizes the
  training locations but does not interpolate to unseen ones.

Note on the two splits' coordinate frames
-----------------------------------------
``DeepMIMODataset`` normalizes positions by *its own file's* maximum, so
``train.mat`` and ``test.mat`` each carry a separate ``scale_factor``.  Each
split is therefore rendered from its own dataset object's normalized positions
-- exactly the tensors ``train.py`` fed the renderer -- and both are asserted to
de-normalize back onto their raw ``.mat`` coordinates before any number is
reported.
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

# Metric plumbing, checkpoint handling and the render path are reused verbatim
# so this script cannot drift away from the numbers the other evals print.
import eval_baseline_rt
import eval_render
from eval_render import EPS
from utils.loss import normalize_mag_map


# ----------------------------------------------------------------------
# Fixed inputs
# ----------------------------------------------------------------------
DATASET_DIR = "dataset/asu_campus_16by64_lt_entire_stride4"
MIMOGS_RUN = "outputs/20260831_084700"
OUTPUT_DIR = os.path.join("analysis", "eval_trainnmse")

# One batch size for BOTH splits -- the comparison is only meaningful if the two
# renders go down an identical code path.  The gain MLP allocates B x N_gaussians
# activations, so this stays small enough to fit alongside a 25k-Gaussian model.
BATCH_SIZE = 32

# Rows are rendered and scored in blocks so a 68k-location split never has to
# materialize all of its predictions at once.
SCORE_CHUNK = 2048

# Below this many dB of separation the two splits count as "the same number".
GAP_THRESHOLD_DB = 1.0

# Reported alongside the headline NMSE; both come out of ``score_prediction``.
TOP1_KEY = "topk_acc_K1"
C4_KEY = "power_capture_K4"

CSV_COLUMNS = (
    "split",
    "n_total",
    "n_scored",
    "n_skipped_zero_power",
    "NMSE_shape_mean_dB",
    "NMSE_shape_median_dB",
    "NMSE_shape_p5_dB",
    "NMSE_shape_p95_dB",
    "NMSE_raw_mean_dB",
    "top1_acc",
    "C4",
)


# ----------------------------------------------------------------------
# Frame check
# ----------------------------------------------------------------------
def check_frame(
    split: str,
    dataset,
    mat_path: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Assert the renderer's positions de-normalize onto the raw ``.mat`` ones.

    Returns ``(raw_positions, raw_magnitude, scale_factor)``.  Exits -- rather
    than reporting numbers from a mismatched frame -- if the round trip drifts.
    """
    raw_positions, raw_magnitude = eval_baseline_rt.load_raw_mat(mat_path)
    scale_factor = float(getattr(dataset, "scale_factor", 1.0))
    normalized = dataset.positions.numpy().astype(np.float64)

    if normalized.shape != raw_positions.shape:
        print(
            f"[eval_trainnmse] FRAME CHECK FAILED for '{split}': the Scene "
            f"dataset holds {normalized.shape} positions but '{mat_path}' holds "
            f"{raw_positions.shape}."
        )
        raise SystemExit(1)

    drift = float(np.max(np.abs(normalized * scale_factor - raw_positions)))
    if not drift < 1e-2:
        print()
        print("-" * 78)
        print(
            f"[eval_trainnmse] FRAME CHECK FAILED for '{split}': positions do "
            f"not de-normalize back onto '{mat_path}' (max drift {drift:.6g} m "
            f">= 1e-2 m)."
        )
        print(f"  scale_factor          : {scale_factor:.10g}")
        print(eval_baseline_rt.describe_ranges("raw .mat", raw_positions))
        print(eval_baseline_rt.describe_ranges("normalized x scale", normalized * scale_factor))
        print(eval_baseline_rt.describe_ranges("normalized (as fed)", normalized))
        print("-" * 78)
        raise SystemExit(1)

    return raw_positions, raw_magnitude, scale_factor


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def score_split(
    split: str,
    dataset,
    scene,
    gaussians,
    model_params,
    device: torch.device,
    use_cuda_rasterizer: bool,
) -> Dict[str, object]:
    """Render and score one split with ``eval_baseline_rt``'s scorer."""
    total = len(dataset)
    collected: Dict[str, List[np.ndarray]] = {}
    skipped_zero_power = 0

    for start in range(0, total, SCORE_CHUNK):
        stop = min(start + SCORE_CHUNK, total)

        ground_truth = dataset.magnitude[start:stop].to(device)
        ground_truth = ground_truth.reshape(
            ground_truth.shape[0], scene.beam_rows, scene.beam_cols
        )

        # Zero-power maps make the NMSE denominator degenerate; drop them and
        # report the count, the same rule ``eval_render.evaluate_test_set`` uses.
        peak = ground_truth.reshape(ground_truth.shape[0], -1).amax(dim=1)
        valid = peak > EPS
        num_valid = int(valid.sum().item())
        skipped_zero_power += int(stop - start) - num_valid
        if num_valid == 0:
            continue

        ground_truth = ground_truth[valid]
        positions = dataset.positions[start:stop][valid.cpu()]

        predicted = eval_baseline_rt.render_mimogs(
            scene,
            gaussians,
            model_params,
            device,
            positions,
            BATCH_SIZE,
            use_cuda_rasterizer,
        )

        scored = eval_baseline_rt.score_prediction(
            predicted, normalize_mag_map(ground_truth)
        )
        for key, values in scored.items():
            collected.setdefault(key, []).append(values)

        done = min(stop, total)
        print(
            f"\r[eval_trainnmse]   {split}: {done}/{total} locations rendered ...",
            end="",
            flush=True,
        )

    print()
    if not collected:
        raise SystemExit(
            f"[eval_trainnmse] Every '{split}' map had zero power; no metric "
            f"could be computed."
        )

    merged = {key: np.concatenate(parts) for key, parts in collected.items()}
    shape_stats = eval_render.summarize(merged["nmse_shape_db"])
    raw_stats = eval_render.summarize(merged["nmse_raw_db"])

    return {
        "split": split,
        "n_total": total,
        "n_scored": int(merged["nmse_shape_db"].size),
        "n_skipped_zero_power": skipped_zero_power,
        "NMSE_shape_mean_dB": shape_stats["mean"],
        "NMSE_shape_median_dB": shape_stats["median"],
        "NMSE_shape_p5_dB": shape_stats["p5"],
        "NMSE_shape_p95_dB": shape_stats["p95"],
        "NMSE_raw_mean_dB": raw_stats["mean"],
        "top1_acc": float(np.mean(merged[TOP1_KEY])),
        "C4": float(np.mean(merged[C4_KEY])),
    }


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
HEADER = (
    f"{'split':<7} {'shape mean':>11} {'median':>9} {'p5':>9} {'p95':>9} "
    f"{'raw mean':>10} {'top-1':>8} {'C4':>8} {'n_scored':>9} {'skipped':>8}"
)


def format_row(row: Dict[str, object]) -> str:
    return (
        f"{row['split']:<7} "
        f"{row['NMSE_shape_mean_dB']:>10.3f}  "
        f"{row['NMSE_shape_median_dB']:>8.3f} "
        f"{row['NMSE_shape_p5_dB']:>8.3f} "
        f"{row['NMSE_shape_p95_dB']:>8.3f} "
        f"{row['NMSE_raw_mean_dB']:>9.3f}  "
        f"{row['top1_acc']:>7.4f} "
        f"{row['C4']:>7.4f} "
        f"{row['n_scored']:>9d} "
        f"{row['n_skipped_zero_power']:>8d}"
    )


def write_summary_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in CSV_COLUMNS})


# ----------------------------------------------------------------------
def main() -> None:
    repository_root = os.path.dirname(os.path.abspath(__file__))

    dataset_root = os.path.join(repository_root, DATASET_DIR)
    if not os.path.isdir(dataset_root):
        dataset_parent = os.path.join(repository_root, "dataset")
        print(f"[eval_trainnmse] Missing dataset directory: {dataset_root}")
        if os.path.isdir(dataset_parent):
            print(f"[eval_trainnmse] Candidates under '{dataset_parent}':")
            for name in sorted(os.listdir(dataset_parent)):
                if os.path.isdir(os.path.join(dataset_parent, name)):
                    print(f"    {name}")
        else:
            print(f"[eval_trainnmse] '{dataset_parent}' does not exist either.")
        raise SystemExit(1)

    outputs_root = os.path.join(repository_root, "outputs")
    run_dir, checkpoint_path = eval_render.resolve_run_dir(
        os.path.join(repository_root, MIMOGS_RUN), outputs_root
    )
    run_name = os.path.basename(os.path.normpath(run_dir))

    print("=" * 78)
    print(f"[eval_trainnmse] RUN        : {run_name}")
    print(f"[eval_trainnmse] checkpoint : {checkpoint_path}")
    print(f"[eval_trainnmse] dataset    : {dataset_root}")
    print("=" * 78)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_params, opt_params = eval_render.restore_config(run_dir, checkpoint)
    model_params.source_path = dataset_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda_rasterizer = (
        bool(int(getattr(model_params, "use_cuda_rasterizer", 1)))
        and device.type == "cuda"
    )

    hidden_dim = eval_render.gain_net_hidden_dim(checkpoint)
    with eval_render.gain_net_width(hidden_dim):
        scene, gaussians = eval_render.build_scene_and_model(
            model_params, opt_params, checkpoint, device
        )

    gain_width = int(gaussians.dynamic_gain_net.net[0].weight.shape[0])

    print()
    print("[eval_trainnmse] CHECKPOINT")
    print(f"  iteration            : {int(checkpoint.get('iteration', -1))}")
    print(f"  gaussians            : {int(gaussians.get_xyz.shape[0])}")
    print(
        f"  gain-MLP hidden width: {gain_width}"
        + ("" if hidden_dim is None else "  (rebuilt to match the checkpoint)")
    )
    print(f"  beam grid            : {scene.beam_rows} x {scene.beam_cols}")
    print(f"  rx array / tx array  : {scene.rx_shape} / {scene.tx_shape}")
    print(
        f"  max_active_rx_beams  : "
        f"{int(getattr(model_params, 'max_active_rx_beams', 2))}"
    )
    print(
        f"  max_active_tx_beams  : "
        f"{int(getattr(model_params, 'max_active_tx_beams', 2))}"
    )
    print(
        f"  device={device} | batch_size={BATCH_SIZE} | "
        f"cuda_rasterizer={int(use_cuda_rasterizer)}"
    )

    splits = (
        ("train", scene.train_set, os.path.join(dataset_root, "train.mat")),
        ("test", scene.test_set, os.path.join(dataset_root, "test.mat")),
    )

    # ------------------------------------------------------------------
    # Frame check -- both splits, before anything is rendered.
    # ------------------------------------------------------------------
    print()
    print("[eval_trainnmse] FRAME CHECK (normalized positions x scale_factor vs. raw .mat)")
    for split, dataset, mat_path in splits:
        raw_positions, raw_magnitude, scale_factor = check_frame(
            split, dataset, mat_path
        )
        drift = float(
            np.max(
                np.abs(
                    dataset.positions.numpy().astype(np.float64) * scale_factor
                    - raw_positions
                )
            )
        )
        # The dataset object and the raw .mat must also be the same rows in the
        # same order, or the positions and the targets would not correspond.
        if not torch.equal(dataset.magnitude, torch.from_numpy(raw_magnitude)):
            print(
                f"[eval_trainnmse] FRAME CHECK FAILED for '{split}': the Scene "
                f"magnitudes differ from '{mat_path}'."
            )
            raise SystemExit(1)
        print(
            f"  {split:<6} N={len(dataset):<7d} scale_factor={scale_factor:12.6f}  "
            f"max drift={drift:.3g} m  OK"
        )

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    print()
    rows = [
        score_split(
            split,
            dataset,
            scene,
            gaussians,
            model_params,
            device,
            use_cuda_rasterizer,
        )
        for split, dataset, _ in splits
    ]

    by_split = {row["split"]: row for row in rows}
    gap_db = (
        by_split["test"]["NMSE_shape_mean_dB"] - by_split["train"]["NMSE_shape_mean_dB"]
    )

    print()
    print("=" * 110)
    print("[eval_trainnmse] SHAPE NMSE (dB, lower is better) -- same checkpoint, same metric code")
    print("-" * 110)
    print(HEADER)
    print("-" * 110)
    for row in rows:
        print(format_row(row))
    print("-" * 110)
    print(
        f"train-test gap in mean shape NMSE : {gap_db:+.3f} dB "
        f"(test {by_split['test']['NMSE_shape_mean_dB']:.3f} - "
        f"train {by_split['train']['NMSE_shape_mean_dB']:.3f})"
    )
    print("=" * 110)

    if abs(gap_db) < GAP_THRESHOLD_DB:
        verdict = (
            f"train ~= test (|gap| {abs(gap_db):.3f} dB < {GAP_THRESHOLD_DB:g} dB): "
            f"FITTING-CAPACITY LIMITED -- the model does not even fit its own "
            f"training locations, so the bottleneck is the representation, not "
            f"generalization."
        )
    elif gap_db >= GAP_THRESHOLD_DB:
        verdict = (
            f"train >> test (gap {gap_db:+.3f} dB): GENERALIZATION GAP -- the "
            f"model fits its training locations markedly better than unseen ones."
        )
    else:
        verdict = (
            f"test better than train by {abs(gap_db):.3f} dB: no generalization "
            f"gap; the test split is simply the easier set of locations."
        )
    print(f"[eval_trainnmse] VERDICT: {verdict}")

    summary_path = os.path.join(repository_root, OUTPUT_DIR, "summary.csv")
    write_summary_csv(summary_path, rows)
    print(f"[eval_trainnmse] wrote {summary_path}")


if __name__ == "__main__":
    main()
