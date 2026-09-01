#!/usr/bin/env python3
"""Score the position-MLP with and without positional encoding.

One question: does the coordinate-MLP baseline lose accuracy when the Fourier
positional encoding is removed?  Both checkpoints are scored on the FULL test
set of ``dataset/asu_campus_16by64_lt`` through exactly the path
``evaluation/eval_t1.py`` uses for its ``MLP`` row::

    ED.TestGroundTruth -> ED.load_mlp -> ED.predict_mlp_maps
                       -> TestGroundTruth.score -> ED.summarize_scores

so the PE row here is the same number the T1 table reports (-26.23 dB shape
NMSE mean); reproducing it to within 0.05 dB is asserted before anything else
is printed.  The learning-free nearest-neighbour row
(``ED.nearest_neighbour_maps`` over the full train split) is rendered for
reference, exactly as ``eval_t1.py`` renders it.

Both MLPs are additionally scored on the TRAIN split.  That row uses the same
``ED.TestGroundTruth`` class unmodified: the class reads ``test.mat`` out of the
directory it is handed, so the train split is presented to it through a scratch
directory whose ``test.mat`` is a symlink to the dataset's ``train.mat``.  The
per-file auto-normalization then lands on the train split's own max|coordinate|,
which is the scale the models were trained with.

Nothing in the repository is modified.  Zero-argument runnable::

    python eval_mlp_nope.py

Writes ``analysis/eval_mlp_nope/{summary.csv, per_location.csv, README.txt}``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
EVALUATION_DIR = os.path.join(REPO_ROOT, "evaluation")
for _entry in (EVALUATION_DIR, REPO_ROOT):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from evaluation import eval_density as ED  # noqa: E402


DATASET_DIR = os.path.join(REPO_ROOT, "dataset", "asu_campus_16by64_lt")
PE_CKPT = os.path.join(REPO_ROOT, "outputs", "density", "MLP", "model_100.pth")
NOPE_CKPT = os.path.join(REPO_ROOT, "outputs", "mlp_nope", "model.pth")
OUTPUT_DIR = os.path.join(REPO_ROOT, "analysis", "eval_mlp_nope")

# The published T1 value for outputs/density/MLP/model_100.pth.
T1_MLP_SHAPE_NMSE_DB = -26.23
T1_TOLERANCE_DB = 0.05

ROW_NN = "Nearest neighbor"
ROW_PE = "MLP (PE, existing)"
ROW_NOPE = "MLP (no PE, new)"


def relative(path: str) -> str:
    try:
        return os.path.relpath(path, REPO_ROOT)
    except ValueError:
        return path


def train_split_ground_truth(dataset_dir: str, device: torch.device, scratch: str):
    """``ED.TestGroundTruth`` over ``train.mat``, via a symlinked scratch dir.

    ``TestGroundTruth.__init__`` reads exactly one file -- ``test.mat`` inside the
    directory it is given (evaluation/eval_density.py:460) -- so pointing that
    name at ``train.mat`` scores the train split through the identical code:
    same zero-power skip rule, same per-file position normalization, same
    ``normalize_mag_map`` target.
    """
    os.makedirs(scratch, exist_ok=True)
    link = os.path.join(scratch, "test.mat")
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.join(os.path.abspath(dataset_dir), "train.mat"), link)
    return ED.TestGroundTruth(scratch, device)


def summarize(scored: Dict[str, np.ndarray]) -> Dict[str, float]:
    """``ED.summarize_scores`` plus the two shape-NMSE tail percentiles."""
    summary = dict(ED.summarize_scores(scored))
    shape = np.asarray(scored["nmse_shape_db"], dtype=np.float64)
    summary["nmse_shape_p5_dB"] = float(np.percentile(shape, 5.0))
    summary["nmse_shape_p95_dB"] = float(np.percentile(shape, 95.0))
    return summary


SUMMARY_FIELDS: Tuple[str, ...] = (
    "method",
    "split",
    "n_scored",
    "parameters",
    "nmse_shape_mean_dB",
    "nmse_shape_median_dB",
    "nmse_shape_p5_dB",
    "nmse_shape_p95_dB",
    "nmse_shape_meanlinear_dB",
    "nmse_raw_mean_dB",
    "nmse_raw_median_dB",
    "topk_acc_K1",
    "topk_acc_K4",
    "topk_acc_K8",
    "power_capture_K1",
    "power_capture_K4",
    "source",
)


def summary_row(
    method: str, split: str, n_scored: int, parameters: Optional[int],
    summary: Dict[str, float], source: str,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "method": method,
        "split": split,
        "n_scored": int(n_scored),
        "parameters": "" if parameters is None else int(parameters),
        "source": source,
    }
    for key in SUMMARY_FIELDS:
        if key in row:
            continue
        row[key] = float(summary[key])
    return row


def score_mlp(path: str, ground_truth, device: torch.device, label: str):
    """``eval_t1.py``'s MLP path, verbatim: load_mlp -> predict -> score."""
    loaded = ED.load_mlp(path, device)
    maps = ED.predict_mlp_maps(loaded, ground_truth.positions_normalized)
    ED.assert_finite_nonnegative(maps, label)
    scored = ground_truth.score(maps)
    del maps
    return loaded, scored


def write_csv(path: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def fmt(value: float, width: int = 9, decimals: int = 3) -> str:
    return f"{value:>{width}.{decimals}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Position-MLP PE ablation scorer")
    parser.add_argument("--dataset", type=str, default=DATASET_DIR)
    parser.add_argument("--pe_checkpoint", type=str, default=PE_CKPT)
    parser.add_argument("--nope_checkpoint", type=str, default=NOPE_CKPT)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    arguments = parser.parse_args()

    dataset_dir = os.path.abspath(arguments.dataset)
    for path in (dataset_dir, arguments.pe_checkpoint, arguments.nope_checkpoint):
        if not os.path.exists(path):
            raise SystemExit(f"[nope-eval] required input is missing: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- test split -----------------------------------------------------------
    ground_truth = ED.TestGroundTruth(dataset_dir, device)
    n_scored = ground_truth.num_scored

    print("=" * 96)
    print("[nope-eval] Position-MLP positional-encoding ablation")
    print("=" * 96)
    print(f"  dataset        : {dataset_dir}")
    print(f"  test locations : {len(ground_truth)} (scored {n_scored}, "
          f"skipped zero-power {ground_truth.num_skipped_zero_power})")
    print(f"  beam grid      : {ground_truth.beam_rows} x {ground_truth.beam_cols}")
    print(f"  device         : {device}")
    print("")

    # -- 1. PE checkpoint, and the T1 reproduction gate ----------------------
    loaded_pe, pe_scored = score_mlp(arguments.pe_checkpoint, ground_truth, device, ROW_PE)
    pe_summary = summarize(pe_scored)
    delta_vs_t1 = pe_summary["nmse_shape_mean_dB"] - T1_MLP_SHAPE_NMSE_DB

    print(f"[nope-eval] T1 reproduction check for {relative(arguments.pe_checkpoint)}")
    print(f"            published T1 shape NMSE mean : {T1_MLP_SHAPE_NMSE_DB:8.3f} dB")
    print(f"            scored here                  : {pe_summary['nmse_shape_mean_dB']:8.3f} dB")
    print(f"            delta                        : {delta_vs_t1:+8.4f} dB "
          f"(tolerance +/-{T1_TOLERANCE_DB})")

    if abs(delta_vs_t1) > T1_TOLERANCE_DB:
        raise SystemExit(
            f"[nope-eval] STOP: the PE checkpoint does not reproduce the T1 value "
            f"({pe_summary['nmse_shape_mean_dB']:.4f} dB vs {T1_MLP_SHAPE_NMSE_DB} dB, "
            f"delta {delta_vs_t1:+.4f} dB > {T1_TOLERANCE_DB} dB)."
        )
    print("            OK")
    print("")

    # -- 2. no-PE checkpoint --------------------------------------------------
    loaded_nope, nope_scored = score_mlp(
        arguments.nope_checkpoint, ground_truth, device, ROW_NOPE
    )
    nope_summary = summarize(nope_scored)

    if int(loaded_nope.arch["num_frequencies"]) != 0:
        raise SystemExit(
            f"[nope-eval] STOP: {relative(arguments.nope_checkpoint)} has "
            f"num_frequencies={loaded_nope.arch['num_frequencies']}, expected 0."
        )

    # -- 3. nearest neighbour (full train split) ------------------------------
    train_positions, train_magnitude = ED.load_train_mat(dataset_dir)
    nn_maps, nn_distance = ED.nearest_neighbour_maps(
        train_positions, train_magnitude, ground_truth.positions_m, device
    )
    ED.assert_finite_nonnegative(nn_maps, ROW_NN)
    nn_scored = ground_truth.score(nn_maps)
    nn_summary = summarize(nn_scored)
    scored_nn_distance = nn_distance[ground_truth.valid_indices]
    mean_nn_distance = float(np.mean(scored_nn_distance))
    del nn_maps

    # -- 4. train-split fit for both MLPs -------------------------------------
    scratch = os.path.join(tempfile.gettempdir(), "mlp_nope_train_split")
    train_gt = train_split_ground_truth(dataset_dir, device, scratch)
    _, pe_train_scored = score_mlp(arguments.pe_checkpoint, train_gt, device, ROW_PE)
    _, nope_train_scored = score_mlp(arguments.nope_checkpoint, train_gt, device, ROW_NOPE)
    pe_train_summary = summarize(pe_train_scored)
    nope_train_summary = summarize(nope_train_scored)
    n_train_scored = train_gt.num_scored
    n_train_total = len(train_gt)
    n_train_skipped = train_gt.num_skipped_zero_power

    # -- 5. per-location delta ------------------------------------------------
    pe_db = np.asarray(pe_scored["nmse_shape_db"], dtype=np.float64)
    nope_db = np.asarray(nope_scored["nmse_shape_db"], dtype=np.float64)
    delta_db = pe_db - nope_db                      # negative => PE is better
    pe_better_fraction = float(np.mean(delta_db < 0.0))

    pe_train_db = np.asarray(pe_train_scored["nmse_shape_db"], dtype=np.float64)
    nope_train_db = np.asarray(nope_train_scored["nmse_shape_db"], dtype=np.float64)
    train_delta_db = pe_train_db - nope_train_db
    pe_better_fraction_train = float(np.mean(train_delta_db < 0.0))

    # -- report ---------------------------------------------------------------
    output_dir = os.path.abspath(arguments.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    table_rows = [
        (ROW_NN, nn_summary, None, "(no learning)"),
        (ROW_PE, pe_summary, loaded_pe.parameter_count, relative(arguments.pe_checkpoint)),
        (ROW_NOPE, nope_summary, loaded_nope.parameter_count, relative(arguments.nope_checkpoint)),
    ]

    header = (f"  {'method':<22}{'mean':>9}{'median':>9}{'p5':>9}{'p95':>9}"
              f"{'Top-1':>9}{'Top-4':>9}{'C4':>9}")
    lines: List[str] = []
    lines.append("Test split -- shape NMSE [dB] and beam-selection metrics")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for name, summary, _params, _source in table_rows:
        lines.append(
            f"  {name:<22}{fmt(summary['nmse_shape_mean_dB'])}"
            f"{fmt(summary['nmse_shape_median_dB'])}"
            f"{fmt(summary['nmse_shape_p5_dB'])}"
            f"{fmt(summary['nmse_shape_p95_dB'])}"
            f"{fmt(summary['topk_acc_K1'], 9, 4)}"
            f"{fmt(summary['topk_acc_K4'], 9, 4)}"
            f"{fmt(summary['power_capture_K4'], 9, 4)}"
        )

    print("")
    for line in lines:
        print(line)
    print("")
    print(f"  mean nearest-train distance : {mean_nn_distance:.4f} m "
          f"(median {float(np.median(scored_nn_distance)):.4f} m)")
    print("")

    print("Train split -- shape NMSE [dB] (fit)")
    print(f"  {'method':<22}{'mean':>9}{'median':>9}{'p5':>9}{'p95':>9}")
    print("  " + "-" * 56)
    for name, summary in ((ROW_PE, pe_train_summary), (ROW_NOPE, nope_train_summary)):
        print(f"  {name:<22}{fmt(summary['nmse_shape_mean_dB'])}"
              f"{fmt(summary['nmse_shape_median_dB'])}"
              f"{fmt(summary['nmse_shape_p5_dB'])}"
              f"{fmt(summary['nmse_shape_p95_dB'])}")
    print("")

    pe_gap = pe_summary["nmse_shape_mean_dB"] - pe_train_summary["nmse_shape_mean_dB"]
    nope_gap = nope_summary["nmse_shape_mean_dB"] - nope_train_summary["nmse_shape_mean_dB"]

    print("Deltas (PE - noPE; negative => PE better)")
    print(f"  test  per-location mean   : {np.mean(delta_db):+8.4f} dB")
    print(f"  test  per-location median : {np.median(delta_db):+8.4f} dB")
    print(f"  test  PE better at        : {100.0 * pe_better_fraction:6.2f} % of locations")
    print(f"  train per-location mean   : {np.mean(train_delta_db):+8.4f} dB")
    print(f"  train per-location median : {np.median(train_delta_db):+8.4f} dB")
    print(f"  train PE better at        : {100.0 * pe_better_fraction_train:6.2f} % of locations")
    print(f"  test-minus-train gap  PE  : {pe_gap:+8.4f} dB")
    print(f"  test-minus-train gap noPE : {nope_gap:+8.4f} dB")
    print(f"  parameters PE / noPE      : {loaded_pe.parameter_count:,} / "
          f"{loaded_nope.parameter_count:,}")
    print("")

    # -- summary.csv ----------------------------------------------------------
    summary_rows = [
        summary_row(ROW_NN, "test", n_scored, None, nn_summary, "(no learning)"),
        summary_row(ROW_PE, "test", n_scored, loaded_pe.parameter_count, pe_summary,
                    relative(arguments.pe_checkpoint)),
        summary_row(ROW_NOPE, "test", n_scored, loaded_nope.parameter_count, nope_summary,
                    relative(arguments.nope_checkpoint)),
        summary_row(ROW_PE, "train", n_train_scored, loaded_pe.parameter_count,
                    pe_train_summary, relative(arguments.pe_checkpoint)),
        summary_row(ROW_NOPE, "train", n_train_scored, loaded_nope.parameter_count,
                    nope_train_summary, relative(arguments.nope_checkpoint)),
    ]
    write_csv(
        os.path.join(output_dir, "summary.csv"),
        SUMMARY_FIELDS,
        [[row[field] for field in SUMMARY_FIELDS] for row in summary_rows],
    )

    # -- per_location.csv -----------------------------------------------------
    valid_positions = ground_truth.valid_positions_m
    per_location_header = (
        "test_index", "x_m", "y_m", "z_m", "nn_distance_m",
        "nmse_shape_db_nn", "nmse_shape_db_pe", "nmse_shape_db_nope",
        "delta_pe_minus_nope_db", "pe_better",
    )
    per_location_rows = [
        [
            int(ground_truth.valid_indices[i]),
            float(valid_positions[i, 0]),
            float(valid_positions[i, 1]),
            float(valid_positions[i, 2]),
            float(scored_nn_distance[i]),
            float(nn_scored["nmse_shape_db"][i]),
            float(pe_db[i]),
            float(nope_db[i]),
            float(delta_db[i]),
            int(delta_db[i] < 0.0),
        ]
        for i in range(n_scored)
    ]
    write_csv(
        os.path.join(output_dir, "per_location.csv"),
        per_location_header,
        per_location_rows,
    )

    # -- README.txt -----------------------------------------------------------
    pe_payload = loaded_pe.payload
    nope_payload = loaded_nope.payload
    readme: List[str] = []
    readme.append("Position-MLP positional-encoding ablation")
    readme.append("=" * 96)
    readme.append("")
    readme.append("Question: does the coordinate-MLP baseline lose accuracy when the Fourier")
    readme.append("positional encoding is removed?  One change, everything else held fixed.")
    readme.append("")
    readme.append("Inputs")
    readme.append("-" * 96)
    readme.append(f"  dataset            {relative(dataset_dir)}")
    readme.append(f"  test locations     {len(ground_truth)} "
                  f"(scored {n_scored}, skipped zero-power {ground_truth.num_skipped_zero_power})")
    readme.append(f"  train locations    {n_train_total} "
                  f"(scored {n_train_scored}, skipped zero-power {n_train_skipped})")
    readme.append(f"  PE checkpoint      {relative(arguments.pe_checkpoint)}")
    readme.append(f"                     arch {pe_payload['arch']}")
    readme.append(f"                     epochs {pe_payload['epochs']}, seed {pe_payload['seed']}, "
                  f"n_train {pe_payload['n_train']}, "
                  f"{loaded_pe.parameter_count:,} parameters, "
                  f"{float(pe_payload['train_seconds']):.1f} s")
    readme.append(f"  no-PE checkpoint   {relative(arguments.nope_checkpoint)}")
    readme.append(f"                     arch {nope_payload['arch']}")
    readme.append(f"                     epochs {nope_payload['epochs']}, seed {nope_payload['seed']}, "
                  f"n_train {nope_payload['n_train']}, "
                  f"{loaded_nope.parameter_count:,} parameters, "
                  f"{float(nope_payload['train_seconds']):.1f} s")
    readme.append("")
    readme.append("Scoring path (identical for every row)")
    readme.append("-" * 96)
    readme.append("  evaluation/eval_density.py: TestGroundTruth -> load_mlp -> predict_mlp_maps")
    readme.append("                              -> TestGroundTruth.score -> summarize_scores")
    readme.append("  Headline metric is the SHAPE NMSE: max-normalized prediction vs max-normalized")
    readme.append("  target, per location in dB.  Zero-power maps are skipped by TestGroundTruth,")
    readme.append("  so every row is scored on the same surviving subset.")
    readme.append("  The train-split rows use the same class on a scratch directory whose")
    readme.append("  test.mat is a symlink to the dataset's train.mat.")
    readme.append("")
    readme.append(f"  T1 reproduction: published {T1_MLP_SHAPE_NMSE_DB:.2f} dB, "
                  f"scored {pe_summary['nmse_shape_mean_dB']:.4f} dB, "
                  f"delta {delta_vs_t1:+.4f} dB (tolerance {T1_TOLERANCE_DB}).")
    readme.append("")
    readme.append("Test split")
    readme.append("-" * 96)
    readme.extend(lines[1:])
    readme.append("")
    readme.append(f"  mean nearest-train distance   {mean_nn_distance:.4f} m")
    readme.append(f"  median nearest-train distance {float(np.median(scored_nn_distance)):.4f} m")
    readme.append("")
    readme.append("Train split (fit)")
    readme.append("-" * 96)
    readme.append(f"  {'method':<22}{'mean':>9}{'median':>9}{'p5':>9}{'p95':>9}")
    for name, summary in ((ROW_PE, pe_train_summary), (ROW_NOPE, nope_train_summary)):
        readme.append(f"  {name:<22}{fmt(summary['nmse_shape_mean_dB'])}"
                      f"{fmt(summary['nmse_shape_median_dB'])}"
                      f"{fmt(summary['nmse_shape_p5_dB'])}"
                      f"{fmt(summary['nmse_shape_p95_dB'])}")
    readme.append("")
    readme.append("Per-location delta, PE minus no-PE (negative => PE better)")
    readme.append("-" * 96)
    readme.append(f"  test  mean   {np.mean(delta_db):+8.4f} dB")
    readme.append(f"  test  median {np.median(delta_db):+8.4f} dB")
    readme.append(f"  test  PE better at {100.0 * pe_better_fraction:.2f} % of locations")
    readme.append(f"  train mean   {np.mean(train_delta_db):+8.4f} dB")
    readme.append(f"  train median {np.median(train_delta_db):+8.4f} dB")
    readme.append(f"  train PE better at {100.0 * pe_better_fraction_train:.2f} % of locations")
    readme.append("")
    readme.append(f"  test-minus-train shape NMSE gap: PE {pe_gap:+.4f} dB, "
                  f"no-PE {nope_gap:+.4f} dB")
    readme.append("")
    readme.append("Files")
    readme.append("-" * 96)
    readme.append("  summary.csv       one row per (method, split)")
    readme.append("  per_location.csv  every scored test location, both MLPs and the NN baseline")
    readme.append("")
    readme.append("Produced by eval_mlp_nope.py; checkpoint by train_mlp_nope.py.")
    readme.append("Nothing tracked in the repository was modified.")

    with open(os.path.join(output_dir, "README.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(readme).rstrip() + "\n")

    print(f"[nope-eval] wrote {relative(output_dir)}/"
          "{summary.csv, per_location.csv, README.txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
