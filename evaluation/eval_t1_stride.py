#!/usr/bin/env python3
"""T1-stride -- Nearest-neighbour vs. MIMO-GS on the stride-4 DeepMIMO split.

Zero-argument runnable::

    python eval_t1_stride.py

A two-row cut of the T1 table (``eval_t1.py``) on
``dataset/asu_campus_16by64_lt_entire_stride4``:

1. ``Nearest neighbor`` -- every test map predicted as the ``train.mat`` map at
                           the nearest train position (3-D Euclidean, ORIGINAL
                           meters, ``scipy.spatial.cKDTree``).  No learning.
2. ``MIMO-GS``          -- ``outputs/20260831_084700/model.pth``, rendered
                           through ``eval_render.render_batch``, i.e. the very
                           ``render_fast`` call ``train.py``'s
                           ``evaluate_full_test_quality`` makes, on positions
                           normalized by ``test.mat``'s own ``scale_factor``
                           exactly the way ``DeepMIMODataset`` does.

Both rows are scored by ``eval_baseline_rt.score_prediction`` -- the scorer
``eval_t1.py`` uses -- against one shared, already max-normalized target, on
one shared mask: a test map whose peak is ``<= EPS`` makes the NMSE
denominator degenerate and is skipped for BOTH methods, the same rule
``eval_render.evaluate_test_set`` applies.

Outputs land in ``analysis/eval_t1_stride/``.  Nothing in the repository is
modified.

``eval_t1.py``'s own helper module (``evaluation/eval_density.py``) is not
importable in this working tree, so the handful of things it would have
supplied -- the shared ground truth, the nearest-neighbour predictor and the
score summary -- are rebuilt here against the same primitives
(``eval_render``'s ``EPS`` / ``render_batch``, ``utils.loss.normalize_mag_map``)
rather than routed through it.
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Import plumbing
# ---------------------------------------------------------------------------
# The eval_* scripts import repo-root packages (``scene``, ``arguments``,
# ``utils``) as top-level modules AND import each other as top-level modules,
# so both directories have to be importable -- the arrangement ``eval_t1.py``
# already relies on.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
EVALUATION_DIR = os.path.join(REPO_ROOT, "evaluation")

for _entry in (EVALUATION_DIR, REPO_ROOT):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import eval_render as ER  # noqa: E402  (path set up above)
from eval_baseline_rt import load_raw_mat, score_prediction  # noqa: E402
from utils.loss import normalize_mag_map  # noqa: E402


# ---------------------------------------------------------------------------
# Fixed inputs -- this is a single-configuration table, nothing is discovered
# ---------------------------------------------------------------------------
DATASET_DIR = "dataset/asu_campus_16by64_lt_entire_stride4"
MIMOGS_RUN = "outputs/20260831_084700"
BATCH_SIZE = 32

OUTPUT_DIR = os.path.join(REPO_ROOT, "analysis", "eval_t1_stride")

# ``DeepMIMODataset`` adds this to max|coordinate| before dividing.
DATASET_EPS = 1e-6
# The position-normalization agreement that is asserted, not assumed.
POSITION_MATCH_TOL = 1e-6

ROW_NN = "Nearest neighbor"
ROW_MIMOGS = "MIMO-GS"
ROW_ORDER: Tuple[str, ...] = (ROW_NN, ROW_MIMOGS)
METHOD_COLUMN_WIDTH = 24

TABLE_COLUMNS: Tuple[str, ...] = (
    "nmse_mean_dB",
    "nmse_median_dB",
    "top1",
    "top4",
    "top8",
    "C4",
)
# True when a LOWER value is better; decides the "*" marking.
LOWER_IS_BETTER: Dict[str, bool] = {
    "nmse_mean_dB": True,
    "nmse_median_dB": True,
    "top1": False,
    "top4": False,
    "top8": False,
    "C4": False,
}
COLUMN_HEADER: Dict[str, str] = {
    "nmse_mean_dB": "NMSE mean [dB]",
    "nmse_median_dB": "NMSE med. [dB]",
    "top1": "Top-1",
    "top4": "Top-4",
    "top8": "Top-8",
    "C4": "C4",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def write_csv(path: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def relative(path: str) -> str:
    try:
        return os.path.relpath(path, REPO_ROOT)
    except ValueError:
        return path


def assert_finite_nonnegative(maps: torch.Tensor, label: str) -> None:
    """Guard a predictor's output before it reaches a metric."""
    if not bool(torch.isfinite(maps).all()):
        raise AssertionError(f"[eval_t1_stride] {label}: output contains non-finite values.")
    minimum = float(maps.min())
    if minimum < 0.0:
        raise AssertionError(
            f"[eval_t1_stride] {label}: output has negative entries (min = {minimum:.6g})."
        )


def summarize_scores(scored: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Flatten one ``score_prediction`` result into the reported scalars."""
    return {
        "nmse_mean_dB": float(np.mean(scored["nmse_shape_db"])),
        "nmse_median_dB": float(np.median(scored["nmse_shape_db"])),
        "top1": float(np.mean(scored["topk_acc_K1"])),
        "top4": float(np.mean(scored["topk_acc_K4"])),
        "top8": float(np.mean(scored["topk_acc_K8"])),
        "C4": float(np.mean(scored["power_capture_K4"])),
        "nmse_meanlinear_dB": ER.mean_linear_db(scored["nmse_shape_db"]),
        "nmse_raw_mean_dB": float(np.mean(scored["nmse_raw_db"])),
        "nmse_raw_median_dB": float(np.median(scored["nmse_raw_db"])),
        "C1": float(np.mean(scored["power_capture_K1"])),
    }


# ---------------------------------------------------------------------------
# Ground truth -- loaded once, shared by both methods
# ---------------------------------------------------------------------------
class TestGroundTruth:
    """``test.mat``: the targets, the mask and the normalized query positions.

    Replicates ``DeepMIMODataset``'s per-file position normalization
    (``max|coordinate| + 1e-6``), so the positions handed to the renderer are
    the ones ``train.py``'s evaluation block iterates over, and applies
    ``eval_render``'s zero-power mask so both methods are scored on the same
    subset against the same ``normalize_mag_map`` target.
    """

    def __init__(self, dataset_dir: str, device: torch.device) -> None:
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.device = device

        positions, magnitude = load_raw_mat(os.path.join(self.dataset_dir, "test.mat"))
        self.positions_m = positions.astype(np.float64)
        self.magnitude = torch.as_tensor(magnitude, dtype=torch.float32, device=device)
        self.beam_rows = int(self.magnitude.shape[1])
        self.beam_cols = int(self.magnitude.shape[2])

        positions_f32 = torch.as_tensor(self.positions_m, dtype=torch.float32)
        self.scale_factor = float(positions_f32.abs().max()) + DATASET_EPS
        self.positions_normalized = (positions_f32 / self.scale_factor).to(device)

        peak = self.magnitude.reshape(self.magnitude.shape[0], -1).amax(dim=1)
        self.valid_mask = (peak > ER.EPS).cpu().numpy()
        self.valid_indices = np.nonzero(self.valid_mask)[0].astype(np.int64)
        self.num_skipped_zero_power = int(self.magnitude.shape[0] - self.valid_indices.size)

        self.target_normalized = normalize_mag_map(
            self.magnitude[torch.as_tensor(self.valid_indices, device=device)]
        )

    def __len__(self) -> int:
        return int(self.magnitude.shape[0])

    @property
    def num_scored(self) -> int:
        return int(self.valid_indices.size)

    def score(self, predicted_full: torch.Tensor) -> Dict[str, np.ndarray]:
        """Score a full-length ``(N, Nr, Nt)`` prediction stack on the mask."""
        selected = predicted_full[
            torch.as_tensor(self.valid_indices, device=predicted_full.device)
        ]
        return score_prediction(selected.float(), self.target_normalized)


# ---------------------------------------------------------------------------
# 1. Nearest-neighbour baseline
# ---------------------------------------------------------------------------
def nearest_neighbour_maps(
    dataset_dir: str, test_positions_m: np.ndarray, device: torch.device
) -> Tuple[torch.Tensor, np.ndarray, int]:
    """Every test map predicted as the nearest ``train.mat`` map.

    Matching happens in ORIGINAL meters, never the per-file normalized
    coordinates.  Returns ``(maps, distance_m, num_train)``.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415 - local, like eval_density

    train_positions, train_magnitude = load_raw_mat(
        os.path.join(dataset_dir, "train.mat")
    )
    distances, indices = cKDTree(np.asarray(train_positions, dtype=np.float64)).query(
        np.asarray(test_positions_m, dtype=np.float64), k=1
    )
    picked = np.ascontiguousarray(train_magnitude[np.asarray(indices, dtype=np.int64)])
    return (
        torch.as_tensor(picked, dtype=torch.float32, device=device),
        np.asarray(distances, dtype=np.float64).reshape(-1),
        int(train_positions.shape[0]),
    )


# ---------------------------------------------------------------------------
# 2. MIMO-GS
# ---------------------------------------------------------------------------
class LoadedMIMOGS:
    """A ``train.py`` run directory rebuilt the ``eval_render`` way."""

    def __init__(self, run_dir: str, dataset_dir: str, device: torch.device) -> None:
        checkpoint_path = os.path.join(run_dir, ER.CHECKPOINT_NAME)
        if not os.path.isfile(checkpoint_path):
            raise SystemExit(
                f"[eval_t1_stride] No '{ER.CHECKPOINT_NAME}' in '{run_dir}'."
            )

        self.run_dir = run_dir
        self.checkpoint_path = checkpoint_path
        self.device = device

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_params, opt_params = ER.restore_config(run_dir, checkpoint)

        # Evaluation always runs against the dataset this script names, whatever
        # path the checkpoint happens to carry.
        trained_on = os.path.abspath(str(getattr(model_params, "source_path", "")))
        model_params.source_path = os.path.abspath(dataset_dir)
        self.trained_on = trained_on

        self.hidden_dim = ER.gain_net_hidden_dim(checkpoint)
        with ER.gain_net_width(self.hidden_dim):
            self.scene, self.gaussians = ER.build_scene_and_model(
                model_params, opt_params, checkpoint, device
            )

        self.model_params = model_params
        self.iteration = int(checkpoint.get("iteration", -1))
        self.use_cuda_rasterizer = (
            bool(int(getattr(model_params, "use_cuda_rasterizer", 1)))
            and device.type == "cuda"
        )

    @property
    def num_gaussians(self) -> int:
        return int(self.gaussians.get_xyz.shape[0])

    def parameter_count(self) -> int:
        """Learnable primitive tensors + the gain MLP.

        ``tie_covariance`` makes the tx-side scaling/rotation the very same
        tensor object as the rx-side one, so tensors are de-duplicated by
        identity rather than counted twice.
        """
        seen: Dict[int, int] = {}
        for name in (
            "_xyz",
            "_xyz_tx",
            "_scaling",
            "_rotation",
            "_scaling_tx",
            "_rotation_tx",
            "_opacity",
        ):
            tensor = getattr(self.gaussians, name, None)
            if torch.is_tensor(tensor):
                seen[id(tensor)] = int(tensor.numel())
        total = int(sum(seen.values()))
        total += int(
            sum(int(p.numel()) for p in self.gaussians.dynamic_gain_net.parameters())
        )
        return total


def render_mimogs_maps(
    loaded: LoadedMIMOGS, normalized_positions: torch.Tensor, batch_size: int
) -> torch.Tensor:
    """``(B,3)`` normalized UE positions -> ``(B, Nr, Nt)`` rendered maps.

    ``ER.render_batch`` is ``train.py``'s ``render_fast`` call, argument for
    argument, so nothing about the render path is re-specified here.
    """
    tx_pos = torch.as_tensor(
        loaded.scene.bs_position, dtype=torch.float32, device=loaded.device
    )
    chunks: List[torch.Tensor] = []
    with torch.no_grad():
        total = int(normalized_positions.shape[0])
        for start in range(0, total, int(batch_size)):
            stop = min(start + int(batch_size), total)
            chunks.append(
                ER.render_batch(
                    normalized_positions[start:stop].to(loaded.device),
                    tx_pos,
                    loaded.gaussians,
                    loaded.scene,
                    loaded.model_params,
                    loaded.use_cuda_rasterizer,
                ).float()
            )
    return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def best_by_column(rows: Dict[str, Dict[str, object]]) -> Dict[str, Optional[str]]:
    best: Dict[str, Optional[str]] = {}
    for column in TABLE_COLUMNS:
        candidates = [
            (method, float(row[column]))
            for method, row in rows.items()
            if row.get(column) is not None
        ]
        if not candidates:
            best[column] = None
            continue
        chooser = min if LOWER_IS_BETTER[column] else max
        best[column] = chooser(candidates, key=lambda item: item[1])[0]
    return best


def format_cell(column: str, value: object) -> str:
    if value is None:
        return "n/a"
    if column in ("nmse_mean_dB", "nmse_median_dB"):
        return f"{float(value):.2f}"
    return f"{float(value):.4f}"


def print_table(
    rows: Dict[str, Dict[str, object]],
    order: Sequence[str],
    best: Dict[str, Optional[str]],
) -> None:
    widths = {column: max(len(COLUMN_HEADER[column]), 14) for column in TABLE_COLUMNS}
    header = f"  {'Method':<{METHOD_COLUMN_WIDTH}}" + "".join(
        f"{COLUMN_HEADER[column]:>{widths[column] + 2}}" for column in TABLE_COLUMNS
    )
    header += f"{'n_scored':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for method in order:
        row = rows.get(method)
        if row is None:
            print(f"  {method:<{METHOD_COLUMN_WIDTH}}  (unavailable)")
            continue
        line = f"  {method:<{METHOD_COLUMN_WIDTH}}"
        for column in TABLE_COLUMNS:
            cell = format_cell(column, row.get(column))
            if best.get(column) == method:
                cell = "*" + cell
            line += f"{cell:>{widths[column] + 2}}"
        line += f"{int(row['n_scored']):>10}"
        print(line)
    print("  " + "-" * (len(header) - 2))
    print("  * = best value in the column.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    dataset_dir = os.path.join(REPO_ROOT, DATASET_DIR)
    run_dir = os.path.join(REPO_ROOT, MIMOGS_RUN)

    print("=" * 84)
    print("[eval_t1_stride] Nearest neighbour vs. MIMO-GS -- stride-4 DeepMIMO split")
    print("=" * 84)

    for path in (dataset_dir, os.path.join(dataset_dir, "train.mat"),
                 os.path.join(dataset_dir, "test.mat"), run_dir):
        if not os.path.exists(path):
            raise SystemExit(f"[eval_t1_stride] Required input is missing: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ground_truth = TestGroundTruth(dataset_dir, device)
    n_test = len(ground_truth)
    n_scored = ground_truth.num_scored

    print(f"[eval_t1_stride] device            : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"[eval_t1_stride] dataset           : {relative(dataset_dir)}")
    print(f"[eval_t1_stride] run               : {relative(run_dir)}")
    print(f"[eval_t1_stride] test locations    : {n_test} (scored {n_scored}, "
          f"skipped zero-power {ground_truth.num_skipped_zero_power})")
    print(f"[eval_t1_stride] beam grid         : "
          f"{ground_truth.beam_rows} x {ground_truth.beam_cols}")
    print(f"[eval_t1_stride] pos_scale (test)  : {ground_truth.scale_factor:.6f}")
    print("")

    rows: Dict[str, Dict[str, object]] = {}

    # -- 1. Nearest neighbour -------------------------------------------
    nn_maps, nn_distance, num_train = nearest_neighbour_maps(
        dataset_dir, ground_truth.positions_m, device
    )
    if tuple(nn_maps.shape[1:]) != (ground_truth.beam_rows, ground_truth.beam_cols):
        raise SystemExit(
            f"[eval_t1_stride] Beam-grid mismatch: train.mat "
            f"{tuple(nn_maps.shape[1:])} vs. test.mat "
            f"{(ground_truth.beam_rows, ground_truth.beam_cols)}."
        )
    assert_finite_nonnegative(nn_maps, ROW_NN)
    nn_scored = ground_truth.score(nn_maps)
    del nn_maps
    rows[ROW_NN] = dict(summarize_scores(nn_scored))
    rows[ROW_NN].update(
        {"method": ROW_NN, "n_scored": n_scored, "source": "(no learning)"}
    )
    print(f"[eval_t1_stride] train locations   : {num_train} (full train.mat)")
    print(f"[eval_t1_stride] {ROW_NN:<17}: "
          f"{rows[ROW_NN]['nmse_mean_dB']:8.3f} dB")

    # -- 2. MIMO-GS ------------------------------------------------------
    loaded = LoadedMIMOGS(run_dir, dataset_dir, device)

    # The renderer must see the positions ``train.py``'s evaluation block
    # iterates -- ``Scene``'s own test set -- so assert the two normalizations
    # agree instead of trusting that they do.
    scene_scale = float(getattr(loaded.scene.test_set, "scale_factor", float("nan")))
    scale_drift = abs(scene_scale - ground_truth.scale_factor)
    assert scale_drift <= POSITION_MATCH_TOL * max(1.0, ground_truth.scale_factor), (
        f"pos_scale disagrees: Scene {scene_scale!r} vs. this script "
        f"{ground_truth.scale_factor!r}"
    )
    position_drift = float(
        (
            loaded.scene.test_set.positions.to(device)
            - ground_truth.positions_normalized
        )
        .abs()
        .max()
    )
    assert position_drift <= POSITION_MATCH_TOL, (
        f"normalized test positions disagree with Scene's by {position_drift:.3g}"
    )

    gs_maps = render_mimogs_maps(
        loaded, ground_truth.positions_normalized, BATCH_SIZE
    )
    assert_finite_nonnegative(gs_maps, ROW_MIMOGS)
    gs_scored = ground_truth.score(gs_maps)
    del gs_maps
    rows[ROW_MIMOGS] = dict(summarize_scores(gs_scored))
    rows[ROW_MIMOGS].update(
        {
            "method": ROW_MIMOGS,
            "n_scored": n_scored,
            "source": relative(loaded.checkpoint_path),
        }
    )
    print(f"[eval_t1_stride] {ROW_MIMOGS:<17}: "
          f"{rows[ROW_MIMOGS]['nmse_mean_dB']:8.3f} dB   "
          f"({loaded.num_gaussians} gaussians, {loaded.parameter_count()} "
          f"parameters, iteration {loaded.iteration}, "
          f"cuda_rasterizer={int(loaded.use_cuda_rasterizer)})")
    print("")

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------
    best = best_by_column(rows)
    print("=" * 84)
    print("[eval_t1_stride] TABLE  (shape NMSE, per-location, dB)")
    print("=" * 84)
    print_table(rows, ROW_ORDER, best)
    print("")

    # ------------------------------------------------------------------
    # Head-to-head
    # ------------------------------------------------------------------
    scored_distance = nn_distance[ground_truth.valid_indices]
    nn_values = nn_scored["nmse_shape_db"]
    gs_values = gs_scored["nmse_shape_db"]
    delta_mean_db = float(np.mean(gs_values)) - float(np.mean(nn_values))
    better_mask = gs_values < nn_values
    better_fraction = float(np.mean(better_mask))

    print("-" * 84)
    print("[eval_t1_stride] NEAREST-TRAIN DISTANCE (scored test locations)")
    print(f"  mean   : {float(np.mean(scored_distance)):9.4f} m")
    print(f"  median : {float(np.median(scored_distance)):9.4f} m")
    print(f"  p95    : {float(np.percentile(scored_distance, 95)):9.4f} m")
    print("")
    print("[eval_t1_stride] MIMO-GS vs. NEAREST NEIGHBOUR")
    print(f"  mean shape NMSE, {ROW_NN:<17}: {float(np.mean(nn_values)):9.4f} dB")
    print(f"  mean shape NMSE, {ROW_MIMOGS:<17}: {float(np.mean(gs_values)):9.4f} dB")
    print(f"  MIMO-GS - NN                        : {delta_mean_db:+9.4f} dB "
          f"({'MIMO-GS is better' if delta_mean_db < 0 else 'nearest neighbour is better'})")
    print(f"  locations where MIMO-GS is better   : "
          f"{int(better_mask.sum())} / {better_mask.size} "
          f"({100.0 * better_fraction:.2f}%)")
    print("-" * 84)
    print("")

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_header = (
        ["method"]
        + list(TABLE_COLUMNS)
        + [
            "n_test",
            "n_scored",
            "nmse_meanlinear_dB",
            "nmse_raw_mean_dB",
            "nmse_raw_median_dB",
            "C1",
            "is_best_nmse_mean",
            "source",
        ]
    )
    summary_rows: List[List[object]] = []
    for method in ROW_ORDER:
        row = rows[method]
        summary_rows.append(
            [method]
            + [f"{float(row[column]):.6f}" for column in TABLE_COLUMNS]
            + [
                int(n_test),
                int(row["n_scored"]),
                f"{float(row['nmse_meanlinear_dB']):.6f}",
                f"{float(row['nmse_raw_mean_dB']):.6f}",
                f"{float(row['nmse_raw_median_dB']):.6f}",
                f"{float(row['C1']):.6f}",
                int(best.get("nmse_mean_dB") == method),
                row["source"],
            ]
        )
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    write_csv(summary_path, summary_header, summary_rows)

    # Per-user rows cover every test location; the unscored ones carry the
    # mask flag and their distance, and "nan" wherever a metric does not exist.
    rank_of_test_row = np.full(n_test, -1, dtype=np.int64)
    rank_of_test_row[ground_truth.valid_indices] = np.arange(n_scored, dtype=np.int64)

    per_user_header = [
        "test_idx",
        "scored",
        "nn_dist_m",
        "nmse_shape_dB_nn",
        "top1_nn",
        "C4_nn",
        "nmse_shape_dB_mimogs",
        "top1_mimogs",
        "C4_mimogs",
    ]
    per_user_rows: List[List[object]] = []
    for index in range(n_test):
        rank = int(rank_of_test_row[index])
        record: List[object] = [
            int(index),
            int(rank >= 0),
            f"{float(nn_distance[index]):.6f}",
        ]
        if rank < 0:
            record.extend(["nan"] * 6)
        else:
            for scored in (nn_scored, gs_scored):
                record.extend(
                    [
                        f"{float(scored['nmse_shape_db'][rank]):.6f}",
                        f"{float(scored['topk_acc_K1'][rank]):.6f}",
                        f"{float(scored['power_capture_K4'][rank]):.6f}",
                    ]
                )
        per_user_rows.append(record)
    per_user_path = os.path.join(OUTPUT_DIR, "per_user.csv")
    write_csv(per_user_path, per_user_header, per_user_rows)

    print(f"[eval_t1_stride] wrote {relative(summary_path)}")
    print(f"[eval_t1_stride] wrote {relative(per_user_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
