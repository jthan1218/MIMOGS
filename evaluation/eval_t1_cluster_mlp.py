#!/usr/bin/env python3
"""T1-cluster + MLP -- Nearest neighbour vs. MIMO-GS vs. the position MLP.

Zero-argument runnable::

    python evaluation/eval_t1_cluster_mlp.py

This is ``eval_t1_cluster.py`` with a third row.  The two existing rows are not
re-implemented: ``eval_t1_cluster`` is imported and its ``TestGroundTruth``,
``nearest_neighbour_maps``, ``LoadedMIMOGS`` and ``render_mimogs_maps`` are
called as-is, so the ``Nearest neighbor`` and ``MIMO-GS`` numbers printed here
come out of exactly the same code path -- and, crucially, the same scorer
(``eval_baseline_rt.score_prediction``), the same already-max-normalized target
and the same zero-power mask (``peak > eval_render.EPS``) -- as before.

The new row:

3. ``Position MLP`` -- ``outputs/cluster/MLP/model_100.pth``, the ``mlp_medium``
                       configuration (hidden 512, depth 6) of
                       ``evaluation/train_MLP.py`` trained on
                       ``dataset/asu_campus_16by64_lt_entire_cluster`` with the
                       same architecture / epochs / hyper-parameters as
                       ``outputs/density/MLP/model_100.pth``.

``PositionMLP`` is rebuilt from the checkpoint's own ``arch`` block and fed
``TestGroundTruth.positions_normalized`` -- ``test.mat``'s positions divided by
``test.mat``'s own ``max|coordinate| + 1e-6``, which is what ``DeepMIMODataset``
hands the model during training, so the query convention is unchanged.  Its
``(B, Nr*Nt)`` output is reshaped to ``(B, Nr, Nt)`` and handed to the shared
``TestGroundTruth.score``; no per-row normalization, clamping or rescaling
happens on the way.

Reported per row: raw NMSE, shape NMSE (both mean and median, per location, in
dB), Top-1 / Top-4 / Top-8 beam-pair accuracy and C4 power capture.

Outputs land in ``analysis/eval_t1_cluster_mlp/``.  Nothing existing is
modified.

Note on paths: ``eval_t1_cluster.py`` computes its ``REPO_ROOT`` as the
directory holding itself, which was right when it sat at the repository root
and is off by one level now that it lives in ``evaluation/``.  This script
derives the repository root as the *parent* of its own directory and resolves
every dataset / run / output path against that, so it runs from anywhere.
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Import plumbing
# ---------------------------------------------------------------------------
# Repo-root packages (``scene``, ``arguments``, ``utils``) and the sibling
# eval_* / train_* modules are both imported as top-level modules, so both
# directories go on the path -- evaluation first, so ``import eval_render``
# inside ``eval_t1_cluster`` resolves.
EVALUATION_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EVALUATION_DIR)

for _entry in (EVALUATION_DIR, REPO_ROOT):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import eval_t1_cluster as T1C  # noqa: E402  (path set up above)
import eval_render as ER  # noqa: E402
from train_MLP import PositionMLP  # noqa: E402


# ---------------------------------------------------------------------------
# Fixed inputs
# ---------------------------------------------------------------------------
# The dataset, the MIMO-GS run and the render batch size are taken from
# eval_t1_cluster rather than restated, so the two scripts cannot drift apart.
DATASET_DIR = T1C.DATASET_DIR
MIMOGS_RUN = T1C.MIMOGS_RUN
BATCH_SIZE = T1C.BATCH_SIZE

MLP_CHECKPOINT = "outputs/cluster/MLP/model_100.pth"
# The raw train_MLP.py run directory, used when the repacked checkpoint above
# is absent.
MLP_RUN_CHECKPOINT = "outputs/cluster/MLP/mlp_medium/model.pth"
MLP_BATCH_SIZE = 512

OUTPUT_DIR = os.path.join(REPO_ROOT, "analysis", "eval_t1_cluster_mlp")

ROW_NN = T1C.ROW_NN
ROW_MIMOGS = T1C.ROW_MIMOGS
ROW_MLP = "Position MLP"
ROW_ORDER = (ROW_NN, ROW_MIMOGS, ROW_MLP)
METHOD_COLUMN_WIDTH = 24

TABLE_COLUMNS = (
    "nmse_raw_mean_dB",
    "nmse_raw_median_dB",
    "nmse_mean_dB",
    "nmse_median_dB",
    "top1",
    "top4",
    "top8",
    "C4",
)
LOWER_IS_BETTER: Dict[str, bool] = {
    "nmse_raw_mean_dB": True,
    "nmse_raw_median_dB": True,
    "nmse_mean_dB": True,
    "nmse_median_dB": True,
    "top1": False,
    "top4": False,
    "top8": False,
    "C4": False,
}
COLUMN_HEADER: Dict[str, str] = {
    "nmse_raw_mean_dB": "raw mean [dB]",
    "nmse_raw_median_dB": "raw med. [dB]",
    "nmse_mean_dB": "shape mean [dB]",
    "nmse_median_dB": "shape med. [dB]",
    "top1": "Top-1",
    "top4": "Top-4",
    "top8": "Top-8",
    "C4": "C4",
}
DB_COLUMNS = frozenset(
    ("nmse_raw_mean_dB", "nmse_raw_median_dB", "nmse_mean_dB", "nmse_median_dB")
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def relative(path: str) -> str:
    try:
        return os.path.relpath(path, REPO_ROOT)
    except ValueError:
        return path


def write_csv(path: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


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
    return f"{float(value):.2f}" if column in DB_COLUMNS else f"{float(value):.4f}"


def print_table(
    rows: Dict[str, Dict[str, object]],
    order: Sequence[str],
    best: Dict[str, Optional[str]],
) -> None:
    widths = {column: max(len(COLUMN_HEADER[column]), 12) for column in TABLE_COLUMNS}
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
    print("  * = best value in the column.  NMSE: lower is better; Top-k / C4: higher is better.")


# ---------------------------------------------------------------------------
# The position-MLP row
# ---------------------------------------------------------------------------
class LoadedMLP:
    """``PositionMLP`` rebuilt from a checkpoint's own ``arch``/``config`` block.

    Both checkpoint layouts written in this repository are accepted: the
    repacked one (``arch`` + ``beam_rows``/``beam_cols``, as
    ``outputs/density/MLP/model_100.pth``) and the raw ``train_MLP.py`` run
    (``config`` carrying the same fields).
    """

    def __init__(self, checkpoint_path: str, device: torch.device) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "arch" in payload:
            arch = dict(payload["arch"])
            beam_rows = int(payload["beam_rows"])
            beam_cols = int(payload["beam_cols"])
            trained_on = str(payload.get("dataset_path", ""))
            epochs = payload.get("epochs")
            trajectory = payload.get("test_trajectory", [])
        elif "config" in payload:
            arch = dict(payload["config"])
            beam_rows = int(arch["beam_rows"])
            beam_cols = int(arch["beam_cols"])
            trained_on = str(payload.get("training", {}).get("source_path", ""))
            epochs = payload.get("training", {}).get("epochs")
            trajectory = payload.get("trajectory", [])
        else:
            raise SystemExit(
                f"[eval_t1_cluster_mlp] '{checkpoint_path}' carries neither an "
                f"'arch' nor a 'config' block; cannot rebuild PositionMLP."
            )

        num_outputs = int(arch["num_outputs"])
        if num_outputs != beam_rows * beam_cols:
            raise SystemExit(
                f"[eval_t1_cluster_mlp] num_outputs {num_outputs} is not "
                f"reshapeable to ({beam_rows}, {beam_cols})."
            )

        model = PositionMLP(
            num_outputs=num_outputs,
            hidden=int(arch["hidden"]),
            depth=int(arch["depth"]),
            num_frequencies=int(arch["num_frequencies"]),
            include_input=bool(arch["include_input"]),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval().to(device)

        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = model
        self.arch = arch
        self.beam_rows = beam_rows
        self.beam_cols = beam_cols
        self.trained_on = trained_on
        self.epochs = int(epochs) if epochs is not None else -1
        self.trajectory = trajectory

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    @torch.no_grad()
    def predict(self, normalized_positions: torch.Tensor, batch_size: int) -> torch.Tensor:
        """``(B,3)`` normalized UE positions -> ``(B, Nr, Nt)`` maps."""
        chunks: List[torch.Tensor] = []
        total = int(normalized_positions.shape[0])
        for start in range(0, total, int(batch_size)):
            stop = min(start + int(batch_size), total)
            batch = normalized_positions[start:stop].to(self.device).reshape(-1, 3)
            chunks.append(
                self.model(batch)
                .reshape(-1, self.beam_rows, self.beam_cols)
                .float()
            )
        return torch.cat(chunks, dim=0)


def resolve_mlp_checkpoint() -> str:
    for candidate in (MLP_CHECKPOINT, MLP_RUN_CHECKPOINT):
        path = os.path.join(REPO_ROOT, candidate)
        if os.path.isfile(path):
            return path
    raise SystemExit(
        "[eval_t1_cluster_mlp] No MLP checkpoint found; looked for "
        f"'{MLP_CHECKPOINT}' and '{MLP_RUN_CHECKPOINT}'."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    dataset_dir = os.path.join(REPO_ROOT, DATASET_DIR)
    run_dir = os.path.join(REPO_ROOT, MIMOGS_RUN)
    mlp_path = resolve_mlp_checkpoint()

    print("=" * 108)
    print("[eval_t1_cluster_mlp] Nearest neighbour vs. MIMO-GS vs. position MLP "
          "-- entire-cluster DeepMIMO split")
    print("=" * 108)

    for path in (dataset_dir, os.path.join(dataset_dir, "train.mat"),
                 os.path.join(dataset_dir, "test.mat"), run_dir):
        if not os.path.exists(path):
            raise SystemExit(f"[eval_t1_cluster_mlp] Required input is missing: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The shared target / mask / query positions -- eval_t1_cluster's own class.
    ground_truth = T1C.TestGroundTruth(dataset_dir, device)
    n_test = len(ground_truth)
    n_scored = ground_truth.num_scored

    print(f"[eval_t1_cluster_mlp] device          : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"[eval_t1_cluster_mlp] dataset         : {relative(dataset_dir)}")
    print(f"[eval_t1_cluster_mlp] MIMO-GS run     : {relative(run_dir)}")
    print(f"[eval_t1_cluster_mlp] MLP checkpoint  : {relative(mlp_path)}")
    print(f"[eval_t1_cluster_mlp] test locations  : {n_test} (scored {n_scored}, "
          f"skipped zero-power {ground_truth.num_skipped_zero_power})")
    print(f"[eval_t1_cluster_mlp] beam grid       : "
          f"{ground_truth.beam_rows} x {ground_truth.beam_cols}")
    print(f"[eval_t1_cluster_mlp] pos_scale (test): {ground_truth.scale_factor:.6f}")
    print("")

    rows: Dict[str, Dict[str, object]] = {}
    scored_by_method: Dict[str, Dict[str, np.ndarray]] = {}

    def record(method: str, scored: Dict[str, np.ndarray], source: str) -> None:
        rows[method] = dict(T1C.summarize_scores(scored))
        rows[method].update(
            {"method": method, "n_scored": n_scored, "source": source}
        )
        scored_by_method[method] = scored

    # -- 1. Nearest neighbour -------------------------------------------
    nn_maps, nn_distance, num_train = T1C.nearest_neighbour_maps(
        dataset_dir, ground_truth.positions_m, device
    )
    if tuple(nn_maps.shape[1:]) != (ground_truth.beam_rows, ground_truth.beam_cols):
        raise SystemExit(
            f"[eval_t1_cluster_mlp] Beam-grid mismatch: train.mat "
            f"{tuple(nn_maps.shape[1:])} vs. test.mat "
            f"{(ground_truth.beam_rows, ground_truth.beam_cols)}."
        )
    T1C.assert_finite_nonnegative(nn_maps, ROW_NN)
    record(ROW_NN, ground_truth.score(nn_maps), "(no learning)")
    del nn_maps
    print(f"[eval_t1_cluster_mlp] train locations : {num_train} (full train.mat)")
    print(f"[eval_t1_cluster_mlp] {ROW_NN:<15}: "
          f"{rows[ROW_NN]['nmse_mean_dB']:8.3f} dB shape / "
          f"{rows[ROW_NN]['nmse_raw_mean_dB']:8.3f} dB raw")

    # -- 2. MIMO-GS ------------------------------------------------------
    loaded_gs = T1C.LoadedMIMOGS(run_dir, dataset_dir, device)

    scene_scale = float(getattr(loaded_gs.scene.test_set, "scale_factor", float("nan")))
    scale_drift = abs(scene_scale - ground_truth.scale_factor)
    assert scale_drift <= T1C.POSITION_MATCH_TOL * max(1.0, ground_truth.scale_factor), (
        f"pos_scale disagrees: Scene {scene_scale!r} vs. this script "
        f"{ground_truth.scale_factor!r}"
    )
    position_drift = float(
        (loaded_gs.scene.test_set.positions.to(device)
         - ground_truth.positions_normalized).abs().max()
    )
    assert position_drift <= T1C.POSITION_MATCH_TOL, (
        f"normalized test positions disagree with Scene's by {position_drift:.3g}"
    )

    gs_maps = T1C.render_mimogs_maps(
        loaded_gs, ground_truth.positions_normalized, BATCH_SIZE
    )
    T1C.assert_finite_nonnegative(gs_maps, ROW_MIMOGS)
    record(ROW_MIMOGS, ground_truth.score(gs_maps), relative(loaded_gs.checkpoint_path))
    del gs_maps
    print(f"[eval_t1_cluster_mlp] {ROW_MIMOGS:<15}: "
          f"{rows[ROW_MIMOGS]['nmse_mean_dB']:8.3f} dB shape / "
          f"{rows[ROW_MIMOGS]['nmse_raw_mean_dB']:8.3f} dB raw   "
          f"({loaded_gs.num_gaussians} gaussians, {loaded_gs.parameter_count()} "
          f"parameters, iteration {loaded_gs.iteration}, "
          f"cuda_rasterizer={int(loaded_gs.use_cuda_rasterizer)})")

    # -- 3. Position MLP -------------------------------------------------
    loaded_mlp = LoadedMLP(mlp_path, device)
    if (loaded_mlp.beam_rows, loaded_mlp.beam_cols) != (
        ground_truth.beam_rows,
        ground_truth.beam_cols,
    ):
        raise SystemExit(
            f"[eval_t1_cluster_mlp] MLP beam grid "
            f"({loaded_mlp.beam_rows}, {loaded_mlp.beam_cols}) does not match "
            f"test.mat ({ground_truth.beam_rows}, {ground_truth.beam_cols})."
        )
    mlp_maps = loaded_mlp.predict(ground_truth.positions_normalized, MLP_BATCH_SIZE)
    T1C.assert_finite_nonnegative(mlp_maps, ROW_MLP)
    record(ROW_MLP, ground_truth.score(mlp_maps), relative(mlp_path))
    del mlp_maps
    print(f"[eval_t1_cluster_mlp] {ROW_MLP:<15}: "
          f"{rows[ROW_MLP]['nmse_mean_dB']:8.3f} dB shape / "
          f"{rows[ROW_MLP]['nmse_raw_mean_dB']:8.3f} dB raw   "
          f"(hidden {loaded_mlp.arch['hidden']}, depth {loaded_mlp.arch['depth']}, "
          f"{loaded_mlp.parameter_count()} parameters, "
          f"{loaded_mlp.epochs} epochs)")
    if loaded_mlp.trained_on:
        print(f"[eval_t1_cluster_mlp] MLP trained on  : "
              f"{relative(os.path.abspath(loaded_mlp.trained_on))}")
    print("")

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------
    best = best_by_column(rows)
    print("=" * 108)
    print("[eval_t1_cluster_mlp] TABLE  (per-location NMSE in dB; Top-k / C4 are fractions)")
    print("=" * 108)
    print_table(rows, ROW_ORDER, best)
    print("")

    # ------------------------------------------------------------------
    # Head-to-head against the two incumbents
    # ------------------------------------------------------------------
    mlp_values = scored_by_method[ROW_MLP]["nmse_shape_db"]
    print("-" * 108)
    print("[eval_t1_cluster_mlp] POSITION MLP vs. THE OTHER ROWS (shape NMSE, per location)")
    for other in (ROW_NN, ROW_MIMOGS):
        other_values = scored_by_method[other]["nmse_shape_db"]
        delta = float(np.mean(mlp_values)) - float(np.mean(other_values))
        better = mlp_values < other_values
        print(f"  vs. {other:<17}: {delta:+8.3f} dB mean "
              f"({'MLP is better' if delta < 0 else other + ' is better'}); "
              f"MLP wins at {int(better.sum())} / {better.size} locations "
              f"({100.0 * float(np.mean(better)):.2f}%)")
    print("-" * 108)
    print("")

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_header = (
        ["method"]
        + list(TABLE_COLUMNS)
        + ["n_test", "n_scored", "nmse_meanlinear_dB", "C1",
           "is_best_nmse_shape_mean", "source"]
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
                f"{float(row['C1']):.6f}",
                int(best.get("nmse_mean_dB") == method),
                row["source"],
            ]
        )
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    write_csv(summary_path, summary_header, summary_rows)

    # Per-user rows cover every test location; unscored ones carry the mask
    # flag and their nearest-train distance, and "nan" wherever a metric does
    # not exist.
    rank_of_test_row = np.full(n_test, -1, dtype=np.int64)
    rank_of_test_row[ground_truth.valid_indices] = np.arange(n_scored, dtype=np.int64)

    tags = {ROW_NN: "nn", ROW_MIMOGS: "mimogs", ROW_MLP: "mlp"}
    per_user_header = ["test_idx", "scored", "nn_dist_m"]
    for method in ROW_ORDER:
        tag = tags[method]
        per_user_header += [
            f"nmse_raw_dB_{tag}", f"nmse_shape_dB_{tag}",
            f"top1_{tag}", f"top4_{tag}", f"top8_{tag}", f"C4_{tag}",
        ]

    per_user_rows: List[List[object]] = []
    for index in range(n_test):
        rank = int(rank_of_test_row[index])
        record_row: List[object] = [
            int(index), int(rank >= 0), f"{float(nn_distance[index]):.6f}"
        ]
        if rank < 0:
            record_row.extend(["nan"] * (6 * len(ROW_ORDER)))
        else:
            for method in ROW_ORDER:
                scored = scored_by_method[method]
                record_row.extend(
                    f"{float(scored[key][rank]):.6f}"
                    for key in ("nmse_raw_db", "nmse_shape_db", "topk_acc_K1",
                                "topk_acc_K4", "topk_acc_K8", "power_capture_K4")
                )
        per_user_rows.append(record_row)
    per_user_path = os.path.join(OUTPUT_DIR, "per_user.csv")
    write_csv(per_user_path, per_user_header, per_user_rows)

    print(f"[eval_t1_cluster_mlp] wrote {relative(summary_path)}")
    print(f"[eval_t1_cluster_mlp] wrote {relative(per_user_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
