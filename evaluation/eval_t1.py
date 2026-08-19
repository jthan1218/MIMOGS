#!/usr/bin/env python3
"""T1 -- main DeepMIMO rendering-accuracy comparison table.

Zero-argument runnable::

    python eval_t1.py

Scores five predictors on the ORIGINAL full test set of
``dataset/asu_campus_16by64_lt`` and writes the paper's main DeepMIMO table to
``analysis/eval_t1/``:

1. ``Sionna RT``              -- the ray-traced maps of
                                 ``dataset/asu_sionna_16by64_lt/full_dataset.mat``,
                                 matched onto the test locations exactly the way
                                 ``eval_baseline_rt.py`` matches them.
2. ``Nearest neighbor``       -- each test map predicted as the train map at the
                                 nearest train position (FULL train set, 3-D
                                 Euclidean, original meters).  No learning.
3. ``MLP``                    -- ``outputs/density/MLP/model_100.pth``.
4. ``MIMO-GS``                -- ``outputs/density/mimogs/model_100.pth``.
5. ``Best separable approx.`` -- for every GROUND-TRUTH test map ``X``, the
                                 non-negative rank-one map ``a b^T`` closest to
                                 ``X``, i.e. the truncated-SVD rank-one fit that
                                 ``eval_marginal_oracle.best_rank1_approximation``
                                 computes.  Reused verbatim, not reimplemented.

Row 5 is an ORACLE: it is built from the ground truth it is scored against, so
it is an upper bound on any method that renders the rx and tx sides separately
-- not a method that could be run at test time.  It is excluded from the
LaTeX bolding and marked with a dagger.

Metrics
-------
Identical to the measured table (T2), and imported rather than reimplemented:
``eval_baseline_rt.score_prediction`` (built on ``eval_render``'s ``EPS`` /
``topk_metrics`` / ``normalize_mag_map``) produces every number.  The headline
metric is the per-location SHAPE NMSE -- max-normalized prediction vs.
max-normalized target -- reported as mean and median in dB, alongside top-1 /
top-4 / top-8 beam-pair overlap accuracy and the top-4 power capture ``C4``.
Raw NMSE and ``C1`` stay in the CSV.

Nothing in the repository is modified.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import sys
import types
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Import plumbing
# ---------------------------------------------------------------------------
# The evaluation scripts import repo-root packages (``scene``, ``arguments``,
# ``utils``) as top-level modules AND import each other as top-level modules,
# so both directories have to be importable -- the arrangement
# ``evaluation/eval_density.py`` already relies on.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
EVALUATION_DIR = os.path.join(REPO_ROOT, "evaluation")

for _entry in (EVALUATION_DIR, REPO_ROOT):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


def _install_eval_db_stub() -> Optional[str]:
    """Make ``eval_marginal_oracle`` importable without ``eval_db_16by64``.

    ``eval_marginal_oracle.py`` starts with ``from eval_db_16by64 import to_db``
    and that module is not in this working tree any more, so the import fails
    before the rank-one math can be reached.  Only the qualitative-figure code
    of that script ever calls ``to_db`` and T1 draws no figures, so a stub that
    raises on use lets the rank-one fit be imported VERBATIM instead of copied.
    Returns a note for the README when the stub had to be installed.
    """
    try:
        if importlib.util.find_spec("eval_db_16by64") is not None:
            return None
    except (ImportError, ValueError):
        pass

    module = types.ModuleType("eval_db_16by64")

    def to_db(*args, **kwargs):
        raise NotImplementedError(
            "eval_db_16by64 is not present in this working tree; eval_t1.py "
            "imports eval_marginal_oracle only for its rank-one fit."
        )

    module.to_db = to_db
    sys.modules["eval_db_16by64"] = module
    return (
        "eval_marginal_oracle.py imports eval_db_16by64, which is absent from "
        "this working tree.  eval_t1.py installs a raise-on-use stub for it so "
        "best_rank1_approximation can be imported verbatim; only that script's "
        "figure code would ever have called the stubbed function."
    )


EVAL_DB_STUB_NOTE = _install_eval_db_stub()

import eval_render as ER  # noqa: E402  (path set up above)
from eval_baseline_rt import (  # noqa: E402
    load_raw_mat,
    match_positions,
    score_prediction,
)
from eval_marginal_oracle import (  # noqa: E402
    best_rank1_approximation,
    rank1_energy_fraction,
)
from evaluation import eval_density as ED  # noqa: E402


# ---------------------------------------------------------------------------
# Fixed inputs -- T1 is a single-configuration table, so nothing is discovered
# ---------------------------------------------------------------------------
DATASET_DIR = os.path.join(REPO_ROOT, "dataset", "asu_campus_16by64_lt")
SIONNA_MAT = os.path.join(
    REPO_ROOT, "dataset", "asu_sionna_16by64_lt", "full_dataset.mat"
)
MIMOGS_CKPT = os.path.join(REPO_ROOT, "outputs", "density", "mimogs", "model_100.pth")
MLP_CKPT = os.path.join(REPO_ROOT, "outputs", "density", "MLP", "model_100.pth")

OUTPUT_DIR = os.path.join(REPO_ROOT, "analysis", "eval_t1")
DENSITY_CSV = os.path.join(REPO_ROOT, "analysis", "eval_density", "density_metrics.csv")

MATCH_TOL_M = 1e-3          # eval_baseline_rt.DEFAULT_MATCH_TOL
DENSITY_TOLERANCE_DB = 0.05  # eval_density.REPACK_TOLERANCE_DB
# ``score_prediction`` clamps the NMSE ratio at 1e-12 before the log, so a row
# sitting on this floor has numerically zero error -- i.e. it "beats" the
# zero-error bound and something is wrong with its target.
CLAMP_FLOOR_DB = -120.0
FLOOR_MARGIN_DB = 0.1
SEPARABLE_SVD_CHUNK = 512
# Wide enough for the longest row label plus its "(bound)" marker.
METHOD_COLUMN_WIDTH = 32

ROW_RT = "Sionna RT"
ROW_NN = "Nearest neighbor"
ROW_MLP = "MLP"
ROW_MIMOGS = "MIMO-GS"
ROW_SEPARABLE = "Best separable approx."
ROW_ORDER: Tuple[str, ...] = (ROW_RT, ROW_NN, ROW_MLP, ROW_MIMOGS, ROW_SEPARABLE)
# Rows built with access to the ground truth they are scored against.  They are
# bounds, not methods, and never take part in the "best" bolding.
ORACLE_ROWS = frozenset({ROW_SEPARABLE})

TABLE_COLUMNS: Tuple[str, ...] = (
    "nmse_mean_dB",
    "nmse_median_dB",
    "top1",
    "top4",
    "top8",
    "C4",
)
# True when a LOWER value is better; decides both the bolding and the ordering.
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
LATEX_HEADER: Dict[str, str] = {
    "nmse_mean_dB": r"NMSE mean [dB]",
    "nmse_median_dB": r"NMSE median [dB]",
    "top1": r"Top-1",
    "top4": r"Top-4",
    "top8": r"Top-8",
    "C4": r"$C_4$",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def write_csv(path: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def write_text(path: str, lines: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(str(line) for line in lines).rstrip() + "\n")


def relative(path: str) -> str:
    try:
        return os.path.relpath(path, REPO_ROOT)
    except ValueError:
        return path


def build_table_row(
    method: str,
    scored: Dict[str, np.ndarray],
    n_test: int,
    n_scored: int,
    source: str,
) -> Dict[str, object]:
    """One table row from one ``score_prediction`` result."""
    summary = ED.summarize_scores(scored)
    return {
        "method": method,
        "nmse_mean_dB": summary["nmse_shape_mean_dB"],
        "nmse_median_dB": summary["nmse_shape_median_dB"],
        "top1": summary["topk_acc_K1"],
        "top4": summary["topk_acc_K4"],
        "top8": summary["topk_acc_K8"],
        "C4": summary["power_capture_K4"],
        "n_test": int(n_test),
        "n_scored": int(n_scored),
        "coverage": float(n_scored) / float(max(n_test, 1)),
        "nmse_meanlinear_dB": summary["nmse_shape_meanlinear_dB"],
        "nmse_raw_mean_dB": summary["nmse_raw_mean_dB"],
        "nmse_raw_median_dB": summary["nmse_raw_median_dB"],
        "C1": summary["power_capture_K1"],
        "source": source,
    }


def separable_maps(ground_truth: torch.Tensor) -> torch.Tensor:
    """The best non-negative rank-one fit of every GT map, chunked.

    ``best_rank1_approximation`` is ``eval_marginal_oracle``'s truncated-SVD
    rank-one fit, imported unchanged: it is the TIGHTEST separable ``a b^T``
    for each map, so it upper-bounds anything that renders the two sides
    independently.
    """
    pieces: List[torch.Tensor] = []
    for start in range(0, int(ground_truth.shape[0]), SEPARABLE_SVD_CHUNK):
        stop = min(start + SEPARABLE_SVD_CHUNK, int(ground_truth.shape[0]))
        pieces.append(best_rank1_approximation(ground_truth[start:stop]))
    return torch.cat(pieces, dim=0)


def rank1_energy(ground_truth: torch.Tensor) -> np.ndarray:
    """``sigma_1^2 / sum sigma_i^2`` per map, chunked; how separable the GT is."""
    pieces: List[np.ndarray] = []
    for start in range(0, int(ground_truth.shape[0]), SEPARABLE_SVD_CHUNK):
        stop = min(start + SEPARABLE_SVD_CHUNK, int(ground_truth.shape[0]))
        pieces.append(rank1_energy_fraction(ground_truth[start:stop]))
    return np.concatenate(pieces, axis=0)


# ---------------------------------------------------------------------------
# Sionna RT
# ---------------------------------------------------------------------------
def evaluate_sionna_rt(
    ground_truth: ED.TestGroundTruth,
    warnings: List[str],
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, object]]:
    """Load, match and score the ray-traced maps the eval_baseline_rt way.

    Returns ``(scored, bookkeeping)``.  ``scored`` is ``None`` when nothing
    matched.  ``bookkeeping['scored_rank']`` holds, for every scored RT
    location, its row index inside the ground truth's scored ordering, so the
    per-location CSV can leave the unmatched locations as NaN.
    """
    gt_test_positions, gt_test_magnitude = load_raw_mat(
        os.path.join(ground_truth.dataset_dir, "test.mat")
    )
    sionna_positions, sionna_magnitude = load_raw_mat(SIONNA_MAT)

    if gt_test_magnitude.shape[1:] != sionna_magnitude.shape[1:]:
        raise SystemExit(
            f"[eval_t1] Beam-grid mismatch: GT {gt_test_magnitude.shape[1:]} vs. "
            f"Sionna {sionna_magnitude.shape[1:]}."
        )

    matched_gt, matched_sionna, match_distance = match_positions(
        gt_test_positions, sionna_positions, MATCH_TOL_M
    )

    num_gt_test = int(gt_test_positions.shape[0])
    num_matched = int(matched_gt.size)
    match_fraction = num_matched / max(num_gt_test, 1)

    print(f"[eval_t1] Sionna RT source     : {relative(SIONNA_MAT)}")
    print(f"[eval_t1]   RT locations       : {int(sionna_positions.shape[0])}")
    print(
        f"[eval_t1]   matched onto test  : {num_matched} / {num_gt_test} "
        f"({100.0 * match_fraction:.2f}%, tolerance {MATCH_TOL_M:g} m)"
    )
    if num_matched:
        print(
            f"[eval_t1]   match distance     : max {match_distance.max():.3g} m / "
            f"mean {match_distance.mean():.3g} m"
        )

    bookkeeping: Dict[str, object] = {
        "num_gt_test": num_gt_test,
        "num_sionna_total": int(sionna_positions.shape[0]),
        "num_matched": num_matched,
        "match_fraction": float(match_fraction),
        "match_tol_m": MATCH_TOL_M,
        "scored_rank": np.empty(0, dtype=np.int64),
    }

    if num_matched == 0:
        warnings.append(
            "WARN Sionna RT: no test location matched the ray-traced set; the RT "
            "row is reported as unavailable."
        )
        return None, bookkeeping

    matched_positions = gt_test_positions[matched_gt]
    drift = float(np.max(np.abs(matched_positions - sionna_positions[matched_sionna])))
    assert drift <= MATCH_TOL_M, (
        f"matched positions disagree by {drift:.4g} m, beyond the "
        f"{MATCH_TOL_M:g} m tolerance"
    )

    # ``TestGroundTruth`` drops zero-power maps; the RT row is scored on the
    # intersection of "matched" and "scored", against the very same targets.
    rank_of_test_row = np.full(num_gt_test, -1, dtype=np.int64)
    rank_of_test_row[ground_truth.valid_indices] = np.arange(
        ground_truth.num_scored, dtype=np.int64
    )
    ranks = rank_of_test_row[matched_gt]
    keep = ranks >= 0
    dropped_zero_power = int(num_matched - int(keep.sum()))
    if dropped_zero_power:
        warnings.append(
            f"WARN Sionna RT: {dropped_zero_power} matched location(s) have a "
            f"zero-power ground-truth map and are excluded, like everywhere else."
        )

    scored_rank = ranks[keep].astype(np.int64)
    bookkeeping["scored_rank"] = scored_rank
    bookkeeping["num_scored"] = int(scored_rank.size)

    # Both predictors must see byte-identical targets; assert it rather than
    # trust the two independent index paths.
    gt_from_mat = torch.from_numpy(gt_test_magnitude[matched_gt[keep]])
    gt_from_loader = ground_truth.magnitude[
        torch.as_tensor(ground_truth.valid_indices[scored_rank], device=ground_truth.device)
    ].cpu()
    assert torch.equal(gt_from_mat, gt_from_loader), (
        "The matched GT magnitudes taken straight from test.mat differ from the "
        "ones the shared TestGroundTruth holds; the RT row would not share the "
        "other rows' target."
    )

    prediction = torch.from_numpy(
        np.ascontiguousarray(sionna_magnitude[matched_sionna[keep]])
    ).to(ground_truth.device)
    ED.assert_finite_nonnegative(prediction, ROW_RT)

    dead = int(
        (prediction.reshape(prediction.shape[0], -1).amax(dim=1) <= ER.EPS).sum().item()
    )
    if dead:
        warnings.append(
            f"WARN Sionna RT: {dead} matched RT map(s) are all-zero (no path "
            f"found); they are scored as-is, which is what the baseline predicts."
        )
    bookkeeping["num_dead_rt_maps"] = dead

    target = ground_truth.target_normalized[
        torch.as_tensor(scored_rank, device=ground_truth.device)
    ]
    return score_prediction(prediction.float(), target), bookkeeping


# ---------------------------------------------------------------------------
# Sanity blocks
# ---------------------------------------------------------------------------
def sanity_against_density(rows: Dict[str, Dict[str, object]]) -> List[str]:
    """MIMO-GS / MLP must reproduce eval_density's fraction-1.0 numbers."""
    lines: List[str] = []
    if not os.path.isfile(DENSITY_CSV):
        lines.append(
            f"  {relative(DENSITY_CSV)} is absent -- the cross-check against "
            f"eval_density could not be run."
        )
        return lines

    reference: Dict[str, float] = {}
    for record in ER.read_csv_rows(DENSITY_CSV):
        fraction = ER._as_float(record.get("fraction"))
        if fraction is None or abs(fraction - 1.0) > 1e-9:
            continue
        value = ER._as_float(record.get("nmse_shape_mean_dB"))
        if value is not None:
            reference[str(record.get("method", "")).strip()] = float(value)

    for method in (ROW_MIMOGS, ROW_MLP):
        expected = reference.get(method)
        if expected is None:
            lines.append(
                f"  {method:<18}: no fraction-1.0 row in {relative(DENSITY_CSV)} "
                f"-- not cross-checked."
            )
            continue
        actual = float(rows[method]["nmse_mean_dB"])
        delta = abs(actual - expected)
        verdict = "ok" if delta <= DENSITY_TOLERANCE_DB else "MISMATCH"
        lines.append(
            f"  {method:<18}: this script {actual:9.4f} dB | eval_density "
            f"{expected:9.4f} dB | delta {delta:.4f} dB "
            f"(tolerance {DENSITY_TOLERANCE_DB:.2f} dB) -> {verdict}"
        )
    return lines


def sanity_zero_error_bound(
    per_location: Dict[str, np.ndarray],
    energy_fraction: np.ndarray,
) -> Tuple[List[str], List[str]]:
    """No row may beat the zero-error bound; nothing else is enforced.

    Returns ``(warnings, notes)``.  A row with partial coverage carries NaN at
    the locations it does not predict -- those are bookkeeping, not values, and
    are skipped.  Landing exactly ON the zero-error floor is legitimate for an
    ORACLE row (a ground-truth map that already IS an outer product is fitted
    exactly by its own rank-one fit), so those become a note; the same thing in
    a real predictor stays a warning.
    """
    warnings: List[str] = []
    notes: List[str] = []

    for method, values in per_location.items():
        scored = ~np.isnan(values)  # NaN marks "this row does not cover here"
        broken = scored & ~np.isfinite(values)
        if bool(broken.any()):
            warnings.append(
                f"WARN {method}: {int(broken.sum())} scored location(s) have a "
                f"non-finite NMSE."
            )

        # The helper clamps the error ratio at 1e-12 before the log, so nothing
        # can print below the floor.  Check anyway -- that is the actual bound.
        beats = scored & np.isfinite(values) & (values < CLAMP_FLOOR_DB - FLOOR_MARGIN_DB)
        if bool(beats.any()):
            warnings.append(
                f"WARN {method}: {int(beats.sum())} location(s) report an NMSE "
                f"below the {CLAMP_FLOOR_DB:.0f} dB clamp floor, i.e. better than "
                f"zero error -- impossible; check the target wiring."
            )

        exact = scored & np.isfinite(values) & (values <= CLAMP_FLOOR_DB + FLOOR_MARGIN_DB)
        if not bool(exact.any()):
            continue
        if method in ORACLE_ROWS:
            separable_gt = int(np.sum(energy_fraction[exact] >= 1.0 - 1e-9))
            notes.append(
                f"  {method}: {int(exact.sum())} location(s) are fitted exactly "
                f"(NMSE on the {CLAMP_FLOOR_DB:.0f} dB clamp floor); "
                f"{separable_gt} of them have a ground-truth map whose rank-one "
                f"energy fraction is 1, i.e. the map already IS an outer product, "
                f"so an exact separable fit is expected there rather than a fault."
            )
            if separable_gt != int(exact.sum()):
                warnings.append(
                    f"WARN {method}: {int(exact.sum()) - separable_gt} location(s) "
                    f"are fitted exactly without the ground truth being rank-one; "
                    f"that combination should not occur."
                )
        else:
            warnings.append(
                f"WARN {method}: {int(exact.sum())} location(s) sit on the "
                f"{CLAMP_FLOOR_DB:.0f} dB clamp floor, i.e. numerically zero error "
                f"-- a predictor without ground-truth access should not reach it, "
                f"so check the target wiring."
            )

    return warnings, notes


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def best_by_column(rows: Dict[str, Dict[str, object]]) -> Dict[str, Optional[str]]:
    """Per column, the best NON-ORACLE row; oracle rows never win a column."""
    best: Dict[str, Optional[str]] = {}
    for column in TABLE_COLUMNS:
        candidates = [
            (method, float(row[column]))
            for method, row in rows.items()
            if method not in ORACLE_ROWS and row.get(column) is not None
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
    header += f"{'n_scored':>10}{'coverage':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for method in order:
        row = rows.get(method)
        if row is None:
            print(f"  {method:<{METHOD_COLUMN_WIDTH}}  (unavailable)")
            continue
        label = method + (" (bound)" if method in ORACLE_ROWS else "")
        line = f"  {label:<{METHOD_COLUMN_WIDTH}}"
        for column in TABLE_COLUMNS:
            cell = format_cell(column, row.get(column))
            if best.get(column) == method:
                cell = "*" + cell
            line += f"{cell:>{widths[column] + 2}}"
        line += f"{int(row['n_scored']):>10}{100.0 * float(row['coverage']):>9.2f}%"
        print(line)
    print("  " + "-" * (len(header) - 2))
    print("  * = best among the non-oracle rows.  (bound) rows see the ground truth.")


def latex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def build_latex(
    rows: Dict[str, Dict[str, object]],
    order: Sequence[str],
    best: Dict[str, Optional[str]],
    rt_bookkeeping: Dict[str, object],
    n_scored_full: int,
) -> List[str]:
    """A booktabs table in the measured table's column style."""
    lines: List[str] = [
        "% T1 -- DeepMIMO (asu_campus_16by64_lt) rendering accuracy.",
        "% Generated by eval_t1.py; do not edit by hand.",
        "% Bold marks the best value in each column among the non-oracle rows.",
        "% The dagger row is a BOUND computed from the ground truth itself, not a",
        "% method that can be run at test time; it is excluded from the bolding.",
        r"\begin{tabular}{l" + "c" * len(TABLE_COLUMNS) + "}",
        r"\toprule",
        "Method & "
        + " & ".join(LATEX_HEADER[column] for column in TABLE_COLUMNS)
        + r" \\",
        r"\midrule",
    ]

    non_oracle = [method for method in order if method not in ORACLE_ROWS]
    oracle = [method for method in order if method in ORACLE_ROWS]

    for method in non_oracle:
        row = rows.get(method)
        label = latex_escape(method)
        if method == ROW_RT:
            label += r"$^{\ddagger}$"
        if row is None:
            lines.append(
                label + " & " + " & ".join(["--"] * len(TABLE_COLUMNS)) + r" \\"
            )
            continue
        cells: List[str] = []
        for column in TABLE_COLUMNS:
            cell = format_cell(column, row.get(column))
            if best.get(column) == method:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(label + " & " + " & ".join(cells) + r" \\")

    if oracle:
        lines.append(r"\midrule")
    for method in oracle:
        row = rows.get(method)
        label = latex_escape(method) + r"$^{\dagger}$"
        cells = [format_cell(column, row.get(column)) for column in TABLE_COLUMNS]
        lines.append(label + " & " + " & ".join(cells) + r" \\")

    matched = int(rt_bookkeeping.get("num_scored", 0) or 0)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"% $^{\dagger}$ Oracle bound: for every ground-truth map the closest",
        r"%   non-negative rank-one (separable) map, i.e. the best any method that",
        r"%   renders the rx and tx sides independently could possibly do.  It is",
        r"%   computed FROM the ground truth and is not a predictor.",
        f"% $^{{\\ddagger}}$ Sionna RT covers {matched} of the {n_scored_full} scored "
        f"test locations",
        r"%   (the ray-traced set does not contain the remaining ones); its row is"
        r" scored on that subset only.",
    ]
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    warnings: List[str] = []
    device = ED.resolve_device()

    print("=" * 78)
    print("[eval_t1] T1 -- DeepMIMO rendering-accuracy comparison table")
    print("=" * 78)

    for path in (DATASET_DIR, MIMOGS_CKPT, MLP_CKPT, SIONNA_MAT):
        if not os.path.exists(path):
            raise SystemExit(f"[eval_t1] Required input is missing: {path}")

    ground_truth = ED.TestGroundTruth(DATASET_DIR, device)
    train_positions, train_magnitude = ED.load_train_mat(DATASET_DIR)

    n_test = len(ground_truth)
    n_scored = ground_truth.num_scored

    print(f"[eval_t1] device               : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"[eval_t1] dataset              : {relative(DATASET_DIR)}")
    print(f"[eval_t1] test locations       : {n_test} "
          f"(scored {n_scored}, skipped zero-power "
          f"{ground_truth.num_skipped_zero_power})")
    print(f"[eval_t1] beam grid            : "
          f"{ground_truth.beam_rows} x {ground_truth.beam_cols}")
    print(f"[eval_t1] train locations      : {int(train_positions.shape[0])} (full set)")
    print("")

    rows: Dict[str, Dict[str, object]] = {}
    per_location: Dict[str, np.ndarray] = {}

    # -- 1. Sionna RT ----------------------------------------------------
    rt_scored, rt_bookkeeping = evaluate_sionna_rt(ground_truth, warnings)
    if rt_scored is not None:
        rows[ROW_RT] = build_table_row(
            ROW_RT,
            rt_scored,
            n_test=n_scored,
            n_scored=int(rt_bookkeeping["num_scored"]),
            source=relative(SIONNA_MAT),
        )
        scattered = np.full(n_scored, np.nan, dtype=np.float64)
        scattered[rt_bookkeeping["scored_rank"]] = rt_scored["nmse_shape_db"]
        per_location[ROW_RT] = scattered
    print("")

    # -- 2. Nearest neighbor --------------------------------------------
    nn_maps, nn_distance = ED.nearest_neighbour_maps(
        train_positions, train_magnitude, ground_truth.positions_m, device
    )
    ED.assert_finite_nonnegative(nn_maps, ROW_NN)
    nn_scored = ground_truth.score(nn_maps)
    rows[ROW_NN] = build_table_row(
        ROW_NN, nn_scored, n_test=n_scored, n_scored=n_scored, source="(no learning)"
    )
    per_location[ROW_NN] = nn_scored["nmse_shape_db"]
    mean_nn_distance = float(np.mean(nn_distance[ground_truth.valid_indices]))
    del nn_maps
    print(f"[eval_t1] {ROW_NN:<22}: {rows[ROW_NN]['nmse_mean_dB']:8.3f} dB   "
          f"(mean nearest-train distance {mean_nn_distance:.4f} m)")

    # -- 3. MLP ----------------------------------------------------------
    loaded_mlp = ED.load_mlp(MLP_CKPT, device)
    mlp_maps = ED.predict_mlp_maps(loaded_mlp, ground_truth.positions_normalized)
    ED.assert_finite_nonnegative(mlp_maps, ROW_MLP)
    mlp_scored = ground_truth.score(mlp_maps)
    rows[ROW_MLP] = build_table_row(
        ROW_MLP, mlp_scored, n_test=n_scored, n_scored=n_scored,
        source=relative(MLP_CKPT),
    )
    per_location[ROW_MLP] = mlp_scored["nmse_shape_db"]
    del mlp_maps
    print(f"[eval_t1] {ROW_MLP:<22}: {rows[ROW_MLP]['nmse_mean_dB']:8.3f} dB   "
          f"(hidden {loaded_mlp.arch['hidden']}, depth {loaded_mlp.arch['depth']}, "
          f"{loaded_mlp.parameter_count} parameters, n_train {loaded_mlp.n_train})")

    # -- 4. MIMO-GS ------------------------------------------------------
    loaded_gs = ED.load_mimogs(MIMOGS_CKPT, device, DATASET_DIR)
    gs_maps = ED.render_mimogs_maps(loaded_gs, ground_truth.positions_normalized)
    ED.assert_finite_nonnegative(gs_maps, ROW_MIMOGS)
    gs_scored = ground_truth.score(gs_maps)
    rows[ROW_MIMOGS] = build_table_row(
        ROW_MIMOGS, gs_scored, n_test=n_scored, n_scored=n_scored,
        source=relative(MIMOGS_CKPT),
    )
    per_location[ROW_MIMOGS] = gs_scored["nmse_shape_db"]
    del gs_maps
    print(f"[eval_t1] {ROW_MIMOGS:<22}: {rows[ROW_MIMOGS]['nmse_mean_dB']:8.3f} dB   "
          f"({loaded_gs.num_gaussians} gaussians, "
          f"{loaded_gs.primitive_parameter_count()} parameters, "
          f"n_train {loaded_gs.n_train})")

    # -- 5. Best separable approximation (ORACLE) ------------------------
    scored_ground_truth = ground_truth.magnitude[
        torch.as_tensor(ground_truth.valid_indices, device=device)
    ]
    separable = separable_maps(scored_ground_truth)
    ED.assert_finite_nonnegative(separable, ROW_SEPARABLE)
    separable_scored = score_prediction(
        separable.float(), ground_truth.target_normalized
    )
    rows[ROW_SEPARABLE] = build_table_row(
        ROW_SEPARABLE,
        separable_scored,
        n_test=n_scored,
        n_scored=n_scored,
        source="(computed from the ground truth)",
    )
    per_location[ROW_SEPARABLE] = separable_scored["nmse_shape_db"]
    energy_fraction = rank1_energy(scored_ground_truth)
    del separable, scored_ground_truth
    print(f"[eval_t1] {ROW_SEPARABLE:<22}: "
          f"{rows[ROW_SEPARABLE]['nmse_mean_dB']:8.3f} dB   "
          f"(rank-one bound, built from the ground truth; mean rank-1 energy "
          f"fraction {float(np.mean(energy_fraction)):.4f})")
    print("")

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------
    best = best_by_column(rows)

    print("=" * 78)
    print("[eval_t1] T1 TABLE  (shape NMSE, per-location, dB)")
    print("=" * 78)
    print_table(rows, ROW_ORDER, best)
    print("")

    # ------------------------------------------------------------------
    # Separable vs. MIMO-GS -- the number that decides the joint-rendering claim
    # ------------------------------------------------------------------
    separable_values = per_location[ROW_SEPARABLE]
    mimogs_values = per_location[ROW_MIMOGS]
    gap_mean_db = float(np.mean(separable_values)) - float(np.mean(mimogs_values))
    worse_mask = separable_values > mimogs_values
    worse_fraction = float(np.mean(worse_mask))
    per_location_gap = separable_values - mimogs_values

    print("-" * 78)
    print("[eval_t1] SEPARABLE BOUND vs. MIMO-GS")
    print(f"  mean shape NMSE, best separable approx. : "
          f"{float(np.mean(separable_values)):9.4f} dB")
    print(f"  mean shape NMSE, MIMO-GS                : "
          f"{float(np.mean(mimogs_values)):9.4f} dB")
    print(f"  gap (separable - MIMO-GS)               : {gap_mean_db:+9.4f} dB "
          f"({'separable is worse' if gap_mean_db > 0 else 'separable is better'} "
          f"on average)")
    print(f"  per-location gap                        : median "
          f"{float(np.median(per_location_gap)):+.4f} dB, p5 "
          f"{float(np.percentile(per_location_gap, 5)):+.4f} dB, p95 "
          f"{float(np.percentile(per_location_gap, 95)):+.4f} dB")
    print(f"  locations where separable is WORSE      : "
          f"{int(worse_mask.sum())} / {separable_values.size} "
          f"({100.0 * worse_fraction:.2f}%)")
    print(f"  mean rank-one energy fraction of the GT : "
          f"{float(np.mean(energy_fraction)):.4f} "
          f"(median {float(np.median(energy_fraction)):.4f})")
    print("-" * 78)
    print("")

    # ------------------------------------------------------------------
    # Ordering -- printed, never enforced
    # ------------------------------------------------------------------
    ordering = sorted(rows.items(), key=lambda item: float(item[1]["nmse_mean_dB"]))
    print("[eval_t1] ORDERING by mean shape NMSE (best first; nothing is enforced)")
    for rank, (method, row) in enumerate(ordering, start=1):
        tag = "  [bound, sees the GT]" if method in ORACLE_ROWS else ""
        coverage = ""
        if int(row["n_scored"]) != n_scored:
            coverage = f"  [on {int(row['n_scored'])} / {n_scored} locations]"
        print(f"  {rank}. {method:<24} {float(row['nmse_mean_dB']):9.4f} dB"
              f"{coverage}{tag}")
    print("")

    # ------------------------------------------------------------------
    # Sanity
    # ------------------------------------------------------------------
    density_lines = sanity_against_density(rows)
    bound_warnings, bound_notes = sanity_zero_error_bound(per_location, energy_fraction)
    warnings.extend(bound_warnings)
    for line in density_lines:
        if "MISMATCH" in line:
            warnings.append("WARN eval_density cross-check failed:" + line.strip())

    print("[eval_t1] SANITY -- fraction 1.0 rows of eval_density")
    for line in density_lines:
        print(line)
    print("[eval_t1] SANITY -- zero-error bound")
    if bound_notes:
        for line in bound_notes:
            print(line)
    else:
        print("  no row reaches the zero-error floor.")
    print("")

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    table_header = (
        ["method"]
        + list(TABLE_COLUMNS)
        + [
            "n_test",
            "n_scored",
            "coverage",
            "nmse_meanlinear_dB",
            "nmse_raw_mean_dB",
            "nmse_raw_median_dB",
            "C1",
            "is_oracle_bound",
            "source",
        ]
    )
    table_rows: List[List[object]] = []
    for method in ROW_ORDER:
        row = rows.get(method)
        if row is None:
            continue
        table_rows.append(
            [method]
            + [f"{float(row[column]):.6f}" for column in TABLE_COLUMNS]
            + [
                int(row["n_test"]),
                int(row["n_scored"]),
                f"{float(row['coverage']):.6f}",
                f"{float(row['nmse_meanlinear_dB']):.6f}",
                f"{float(row['nmse_raw_mean_dB']):.6f}",
                f"{float(row['nmse_raw_median_dB']):.6f}",
                f"{float(row['C1']):.6f}",
                int(method in ORACLE_ROWS),
                row["source"],
            ]
        )
    table_path = os.path.join(OUTPUT_DIR, "t1_table.csv")
    write_csv(table_path, table_header, table_rows)

    # Per-location shape NMSE of every row.
    positions = ground_truth.valid_positions_m
    location_header = ["test_index", "x_m", "y_m", "z_m"] + [
        f"nmse_shape_dB_{method}" for method in ROW_ORDER if method in rows
    ]
    location_rows: List[List[object]] = []
    present = [method for method in ROW_ORDER if method in rows]
    for index in range(n_scored):
        record: List[object] = [
            int(ground_truth.valid_indices[index]),
            f"{float(positions[index, 0]):.6f}",
            f"{float(positions[index, 1]):.6f}",
            f"{float(positions[index, 2]):.6f}",
        ]
        for method in present:
            value = float(per_location[method][index])
            # Literal "nan" rather than a blank cell: the RT row simply has no
            # prediction at these locations, and every CSV reader parses it.
            record.append("nan" if not np.isfinite(value) else f"{value:.6f}")
        location_rows.append(record)
    location_path = os.path.join(OUTPUT_DIR, "t1_per_location.csv")
    write_csv(location_path, location_header, location_rows)

    # The separable row on its own, so its spread can be inspected directly.
    separable_header = [
        "test_index",
        "x_m",
        "y_m",
        "z_m",
        "nmse_shape_dB_separable",
        "nmse_raw_dB_separable",
        "topk_acc_K1_separable",
        "topk_acc_K4_separable",
        "topk_acc_K8_separable",
        "power_capture_K4_separable",
        "rank1_energy_fraction",
        "nmse_shape_dB_MIMO-GS",
        "gap_separable_minus_mimogs_dB",
        "separable_worse_than_mimogs",
    ]
    separable_rows: List[List[object]] = []
    for index in range(n_scored):
        separable_rows.append(
            [
                int(ground_truth.valid_indices[index]),
                f"{float(positions[index, 0]):.6f}",
                f"{float(positions[index, 1]):.6f}",
                f"{float(positions[index, 2]):.6f}",
                f"{float(separable_scored['nmse_shape_db'][index]):.6f}",
                f"{float(separable_scored['nmse_raw_db'][index]):.6f}",
                f"{float(separable_scored['topk_acc_K1'][index]):.6f}",
                f"{float(separable_scored['topk_acc_K4'][index]):.6f}",
                f"{float(separable_scored['topk_acc_K8'][index]):.6f}",
                f"{float(separable_scored['power_capture_K4'][index]):.6f}",
                f"{float(energy_fraction[index]):.6f}",
                f"{float(mimogs_values[index]):.6f}",
                f"{float(per_location_gap[index]):.6f}",
                int(bool(worse_mask[index])),
            ]
        )
    separable_path = os.path.join(OUTPUT_DIR, "t1_separable_per_location.csv")
    write_csv(separable_path, separable_header, separable_rows)

    latex_path = os.path.join(OUTPUT_DIR, "t1_table.tex")
    write_text(latex_path, build_latex(rows, ROW_ORDER, best, rt_bookkeeping, n_scored))

    readme_path = os.path.join(OUTPUT_DIR, "README.txt")
    write_text(
        readme_path,
        build_readme(
            rows=rows,
            per_location=per_location,
            ground_truth=ground_truth,
            train_positions=train_positions,
            rt_bookkeeping=rt_bookkeeping,
            density_lines=density_lines,
            bound_notes=bound_notes,
            warnings=warnings,
            device=device,
            loaded_mlp=loaded_mlp,
            loaded_gs=loaded_gs,
            mean_nn_distance=mean_nn_distance,
            energy_fraction=energy_fraction,
            gap_mean_db=gap_mean_db,
            worse_fraction=worse_fraction,
            per_location_gap=per_location_gap,
            best=best,
        ),
    )

    print(f"[eval_t1] wrote {relative(table_path)}")
    print(f"[eval_t1] wrote {relative(location_path)}")
    print(f"[eval_t1] wrote {relative(separable_path)}")
    print(f"[eval_t1] wrote {relative(latex_path)}")
    print(f"[eval_t1] wrote {relative(readme_path)}")
    print("")

    print("[eval_t1] WARNINGS")
    if warnings:
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("  none")
    print("")
    print("[eval_t1] rerun: python eval_t1.py")
    return 0


def build_readme(
    *,
    rows: Dict[str, Dict[str, object]],
    per_location: Dict[str, np.ndarray],
    ground_truth: ED.TestGroundTruth,
    train_positions: np.ndarray,
    rt_bookkeeping: Dict[str, object],
    density_lines: Sequence[str],
    bound_notes: Sequence[str],
    warnings: Sequence[str],
    device: torch.device,
    loaded_mlp: "ED.LoadedMLP",
    loaded_gs: "ED.LoadedMIMOGS",
    mean_nn_distance: float,
    energy_fraction: np.ndarray,
    gap_mean_db: float,
    worse_fraction: float,
    per_location_gap: np.ndarray,
    best: Dict[str, Optional[str]],
) -> List[str]:
    n_scored = ground_truth.num_scored
    matched = int(rt_bookkeeping.get("num_scored", 0) or 0)
    match_fraction = float(rt_bookkeeping.get("match_fraction", 0.0))

    lines: List[str] = [
        "eval_t1 -- main DeepMIMO rendering-accuracy comparison table (T1)",
        "=" * 78,
        "",
        "DATASET",
        f"  Directory        : {DATASET_DIR}",
        f"  Train locations  : {int(train_positions.shape[0])} (full training set)",
        f"  Test locations   : {len(ground_truth)} "
        f"({n_scored} scored, {ground_truth.num_skipped_zero_power} skipped for "
        f"zero power)",
        f"  Map size         : {ground_truth.beam_rows} x {ground_truth.beam_cols} "
        f"beam pairs (Rx x Tx)",
        "  Split            : the prebaked train.mat / test.mat pair; no random",
        "                     splitting happens anywhere in this script.",
        f"  Mean nearest-train distance of the scored test set: "
        f"{mean_nn_distance:.4f} m",
        "",
        "ROWS AND THEIR SOURCES",
        "  Sionna RT              Ray-traced maps read from",
        f"                         {relative(SIONNA_MAT)}",
        "                         and matched onto the test locations with the",
        "                         greedy one-to-one matcher of eval_baseline_rt.py",
        f"                         (3-D Euclidean, original meters, tolerance "
        f"{MATCH_TOL_M:g} m).",
        "  Nearest neighbor       Each test map predicted as the train map at the",
        "                         nearest train position (FULL train set, 3-D",
        "                         Euclidean, original meters).  No learning.",
        f"  MLP                    {relative(MLP_CKPT)}",
        f"                         PositionMLP hidden={loaded_mlp.arch['hidden']} "
        f"depth={loaded_mlp.arch['depth']} "
        f"outputs={loaded_mlp.arch['num_outputs']} "
        f"(PE num_frequencies={loaded_mlp.arch['num_frequencies']}, "
        f"include_input={loaded_mlp.arch['include_input']}),",
        f"                         {loaded_mlp.parameter_count} parameters, "
        f"n_train={loaded_mlp.n_train}.  Rebuilt from the self-contained repack",
        "                         dict alone (state_dict + arch), no run dir, via",
        "                         evaluation/train_MLP.PositionMLP.",
        f"  MIMO-GS                {relative(MIMOGS_CKPT)}",
        f"                         {loaded_gs.num_gaussians} gaussians, "
        f"{loaded_gs.primitive_parameter_count()} primitive+gain parameters,",
        f"                         n_train={loaded_gs.n_train}.  Rebuilt from the",
        "                         repack's own config block (model_params /",
        "                         opt_params / capture), no run dir.",
        "  Best separable approx. NOT A METHOD -- an oracle bound.  See below.",
        "",
        "THE SEPARABLE ROW IS AN ORACLE (ground-truth access)",
        "  For EVERY ground-truth test map X, this row reports the closest",
        "  non-negative separable map a b^T, i.e. the truncated-SVD rank-one fit",
        "  computed by eval_marginal_oracle.best_rank1_approximation (imported",
        "  verbatim; the Perron-Frobenius sign argument in that function is what",
        "  makes the magnitudes safe to take).  Because a and b are fitted TO the",
        "  ground truth it is scored against, this row cannot be produced at test",
        "  time by any predictor.  It is an UPPER BOUND on every method that",
        "  renders the rx side and the tx side separately and multiplies them:",
        "  no such method can fit X better than X's own best rank-one fit.  For",
        "  that reason it is excluded from the 'best value' bolding in the LaTeX",
        "  table and carries a dagger there.",
        f"  Mean rank-one energy fraction of the ground truth "
        f"(sigma_1^2 / sum sigma_i^2): {float(np.mean(energy_fraction)):.4f} "
        f"(median {float(np.median(energy_fraction)):.4f}).",
        "  The closer that is to 1, the more separable the true maps are and the",
        "  less a joint renderer can gain in principle.",
        "",
        "METRICS (identical to the measured table T2, imported not reimplemented)",
        "  Shape NMSE  : max-normalized prediction vs. max-normalized target,",
        "                per location, in dB.  Reported as mean and median.  This",
        "                is the only one of eval_render's two conventions that is",
        "                comparable across methods -- the raw convention penalises",
        "                any predictor that does not carry the target's",
        "                normalization, which Sionna RT and the nearest-neighbour",
        "                baseline do not.",
        "  Top-1/4/8   : overlap between the K strongest predicted beam pairs and",
        "                the K strongest ground-truth beam pairs, divided by K.",
        "  C4          : power capture at K=4 -- the ground-truth power in the 4",
        "                predicted-best beam pairs, over the power in the 4",
        "                genie-best ones.",
        "  Also in CSV : raw/scale NMSE, mean-linear shape NMSE, C1.",
        "  Provenance  : every number comes from",
        "                eval_baseline_rt.score_prediction, which is built on",
        "                eval_render's EPS / topk_metrics / normalize_mag_map.",
        "  Guards      : every predictor's output is asserted finite and",
        "                non-negative before it reaches a metric",
        "                (eval_density.assert_finite_nonnegative), and the RT row's",
        "                targets are asserted bit-identical to the shared ones.",
        f"  Device      : {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
        "",
        "SIONNA RT COVERAGE",
        f"  Ray-traced locations available : "
        f"{int(rt_bookkeeping.get('num_sionna_total', 0))}",
        f"  Matched onto the test set      : "
        f"{int(rt_bookkeeping.get('num_matched', 0))} / "
        f"{int(rt_bookkeeping.get('num_gt_test', 0))} "
        f"({100.0 * match_fraction:.2f}%)",
        f"  Scored (matched and non-zero-power GT) : {matched} / {n_scored}",
        f"  All-zero RT maps among them    : "
        f"{int(rt_bookkeeping.get('num_dead_rt_maps', 0))} (scored as predicted;",
        "                                   a no-path RT result is a real",
        "                                   prediction, not a missing value)",
        "  The RT row is therefore scored on a SUBSET of the locations every other",
        "  row is scored on, and its per-location column in t1_per_location.csv is",
        "  'nan' at the locations the ray-traced set does not contain.  Read the",
        "  RT row as 'RT on the locations RT covers', not as a same-sample",
        "  comparison.",
        "",
        "TABLE",
    ]

    widths = {column: max(len(COLUMN_HEADER[column]), 14) for column in TABLE_COLUMNS}
    header = f"  {'Method':<{METHOD_COLUMN_WIDTH}}" + "".join(
        f"{COLUMN_HEADER[column]:>{widths[column] + 2}}" for column in TABLE_COLUMNS
    )
    header += f"{'n_scored':>10}"
    lines.append(header)
    for method in ROW_ORDER:
        row = rows.get(method)
        if row is None:
            lines.append(f"  {method:<{METHOD_COLUMN_WIDTH}}  (unavailable)")
            continue
        label = method + (" (bound)" if method in ORACLE_ROWS else "")
        line = f"  {label:<{METHOD_COLUMN_WIDTH}}"
        for column in TABLE_COLUMNS:
            cell = format_cell(column, row.get(column))
            if best.get(column) == method:
                cell = "*" + cell
            line += f"{cell:>{widths[column] + 2}}"
        line += f"{int(row['n_scored']):>10}"
        lines.append(line)
    lines.append(
        "  * = best in that column among the non-oracle rows; (bound) rows are "
        "computed from the ground truth."
    )
    lines.append("")

    ordering = sorted(rows.items(), key=lambda item: float(item[1]["nmse_mean_dB"]))
    lines.append("ORDERING by mean shape NMSE (best first; nothing is enforced)")
    for rank, (method, row) in enumerate(ordering, start=1):
        tag = "  [bound, sees the GT]" if method in ORACLE_ROWS else ""
        coverage = ""
        if int(row["n_scored"]) != n_scored:
            coverage = f"  [on {int(row['n_scored'])} / {n_scored} locations]"
        lines.append(
            f"  {rank}. {method:<24} {float(row['nmse_mean_dB']):9.4f} dB"
            f"{coverage}{tag}"
        )
    lines.append("")

    separable_values = per_location[ROW_SEPARABLE]
    mimogs_values = per_location[ROW_MIMOGS]
    lines += [
        "SEPARABLE BOUND vs. MIMO-GS",
        f"  mean shape NMSE, best separable approx. : "
        f"{float(np.mean(separable_values)):9.4f} dB",
        f"  mean shape NMSE, MIMO-GS                : "
        f"{float(np.mean(mimogs_values)):9.4f} dB",
        f"  gap (separable - MIMO-GS)               : {gap_mean_db:+9.4f} dB",
        f"  per-location gap                        : median "
        f"{float(np.median(per_location_gap)):+.4f} dB, p5 "
        f"{float(np.percentile(per_location_gap, 5)):+.4f} dB, p95 "
        f"{float(np.percentile(per_location_gap, 95)):+.4f} dB",
        f"  locations where the separable fit is WORSE than MIMO-GS : "
        f"{int(np.sum(separable_values > mimogs_values))} / "
        f"{separable_values.size} ({100.0 * worse_fraction:.2f}%)",
        "  No ordering between these two rows is assumed anywhere in this script;",
        "  the numbers above are reported as measured.",
        "",
        "HEADLINE NUMBERS (mean shape NMSE [dB] / top-1 / C4)",
    ]
    for method in ROW_ORDER:
        row = rows.get(method)
        if row is None:
            lines.append(f"  {method:<24} unavailable")
            continue
        suffix = "   (oracle bound)" if method in ORACLE_ROWS else ""
        if int(row["n_scored"]) != n_scored:
            suffix += f"   (on {int(row['n_scored'])} / {n_scored} locations)"
        lines.append(
            f"  {method:<24} {float(row['nmse_mean_dB']):8.3f} dB    "
            f"{float(row['top1']):.4f}    {float(row['C4']):.4f}{suffix}"
        )
    lines.append("")

    lines.append("SANITY")
    lines.append(
        f"  MIMO-GS and MLP must reproduce the fraction-1.0 rows of "
        f"{relative(DENSITY_CSV)}"
    )
    lines.append(f"  to within {DENSITY_TOLERANCE_DB:.2f} dB:")
    lines.extend(density_lines)
    lines.append(
        f"  Zero-error bound: no row may report an NMSE below the "
        f"{CLAMP_FLOOR_DB:.0f} dB clamp floor of the NMSE helper, and no row"
    )
    lines.append(
        "  without ground-truth access may even reach it.  Reaching it IS allowed"
    )
    lines.append("  for the oracle row, and is cross-checked against the GT's rank:")
    if bound_notes:
        lines.extend(bound_notes)
    else:
        lines.append("    no row reaches the zero-error floor.")
    lines.append("  Nothing else is enforced -- in particular no ordering between the")
    lines.append("  rows, and no relation between the separable bound and MIMO-GS.")
    if EVAL_DB_STUB_NOTE:
        lines.append("")
        lines.append("IMPORT NOTE")
        for chunk in EVAL_DB_STUB_NOTE.split(".  "):
            lines.append(f"  {chunk.strip().rstrip('.')}.")
    lines.append("")

    lines += [
        "FILES",
        "  t1_table.csv                   the table above, one row per method",
        "  t1_per_location.csv            per-location shape NMSE of every row plus",
        "                                 x, y, z; the Sionna RT column is 'nan'",
        "                                 where the ray-traced set has no match",
        "  t1_separable_per_location.csv  the separable bound on its own, with the",
        "                                 rank-one energy fraction and the",
        "                                 per-location gap against MIMO-GS",
        "  t1_table.tex                   LaTeX snippet in the measured table's",
        "                                 column style; bold = best among the",
        "                                 non-oracle rows, dagger = the bound",
        "  README.txt                     this file",
        "",
        "WARNINGS",
    ]
    if warnings:
        lines.extend(f"  {warning}" for warning in warnings)
    else:
        lines.append("  none")
    lines += [
        "",
        "RERUN",
        "  python eval_t1.py",
    ]
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
