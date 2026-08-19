"""Per-location shape-NMSE CDF figure (paper T1).

Pure post-processing: this script reads the per-location CSV that
``analysis/eval_t1`` already contains and draws one empirical CDF per method.
Nothing is loaded, rendered or re-scored here -- if the numbers move, they
moved upstream.

    python eval_cdf.py

Inputs
    analysis/eval_t1/t1_per_location.csv   (required)
    analysis/eval_t1/t1_table.csv          (optional, used as a cross-check)

Outputs
    analysis/eval_cdf/fig_nmse_cdf.png
    analysis/eval_cdf/fig_nmse_cdf.pdf
"""

import csv
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Paths and figure conventions (kept identical to the net-rate figures)
# ----------------------------------------------------------------------
PER_LOCATION_CSV = os.path.join("analysis", "eval_t1", "t1_per_location.csv")
TABLE_CSV = os.path.join("analysis", "eval_t1", "t1_table.csv")
OUTPUT_DIR = os.path.join("analysis", "eval_cdf")
FIGURE_STEM = "fig_nmse_cdf"

FIGURE_DPI = 300
LABEL_FONTSIZE = 14
TICK_LABELSIZE = 12
LEGEND_FONTSIZE = 10

# Legend order is the drawing order.  MIMO-GS keeps "tab:blue" and MLP keeps
# "tab:brown" so the reader can carry the colours over from the net-rate
# figures of eval_net_rate.py; the remaining two are simply well separated.
# (scheme label, csv column, colour, linewidth, zorder)
CDF_SERIES = (
    ("MIMO-GS", "nmse_shape_dB_MIMO-GS", "tab:blue", 1.8, 5),
    ("MLP", "nmse_shape_dB_MLP", "tab:brown", 1.6, 4),
    ("Nearest neighbor", "nmse_shape_dB_Nearest neighbor", "tab:purple", 1.6, 3),
    ("Sionna RT", "nmse_shape_dB_Sionna RT", "tab:red", 1.6, 2),
)

# The x window is the union of the per-method [1st, 99th] percentile spans,
# padded by this fraction of its width on each side.
PERCENTILE_LOW = 1.0
PERCENTILE_HIGH = 99.0
X_PADDING_FRACTION = 0.04

# A mean recomputed from the per-location CSV that differs from the published
# table by more than this many dB means the two files disagree.
MEAN_TOLERANCE_DB = 0.05


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def read_per_location(path: str) -> Dict[str, np.ndarray]:
    """Read the required columns, NaN for blank/non-numeric cells."""
    if not os.path.isfile(path):
        raise SystemExit(
            f"[eval_cdf] required input is missing: {path}\n"
            f"[eval_cdf] this script only post-processes existing results; "
            f"run the T1 evaluation first so that {path} exists."
        )

    wanted = [column for _, column, _, _, _ in CDF_SERIES]
    columns: Dict[str, List[float]] = {column: [] for column in wanted}

    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [column for column in wanted if column not in header]
        if missing:
            raise SystemExit(
                f"[eval_cdf] {path} does not carry the expected columns: "
                f"{missing}\n[eval_cdf] found: {header}"
            )
        for row in reader:
            for column in wanted:
                text = (row.get(column) or "").strip()
                try:
                    columns[column].append(float(text))
                except ValueError:
                    columns[column].append(float("nan"))

    values = {column: np.asarray(items, dtype=np.float64)
              for column, items in columns.items()}
    total = next(iter(values.values())).size if values else 0
    if total == 0:
        raise SystemExit(f"[eval_cdf] {path} has a header but no data rows.")
    return values


def read_table_means(path: str) -> Dict[str, Dict[str, float]]:
    """Published mean/median per method; empty dict if the table is absent."""
    if not os.path.isfile(path):
        return {}
    published: Dict[str, Dict[str, float]] = {}
    with open(path, "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            method = (row.get("method") or "").strip()
            if not method:
                continue
            entry: Dict[str, float] = {}
            for key in ("nmse_mean_dB", "nmse_median_dB"):
                try:
                    entry[key] = float(row.get(key, ""))
                except (TypeError, ValueError):
                    continue
            if entry:
                published[method] = entry
    return published


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
def save_figure(figure, output_dir: str, stem: str) -> None:
    """Write ``<stem>.png`` (300 dpi) and ``<stem>.pdf`` into ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, f"{stem}.png"), dpi=FIGURE_DPI)
    figure.savefig(os.path.join(output_dir, f"{stem}.pdf"))
    plt.close(figure)


def x_limits(finite: Sequence[np.ndarray]) -> Optional[Sequence[float]]:
    """Union of the per-method [1st, 99th] percentile spans, padded."""
    lows, highs = [], []
    for values in finite:
        if values.size == 0:
            continue
        lows.append(float(np.percentile(values, PERCENTILE_LOW)))
        highs.append(float(np.percentile(values, PERCENTILE_HIGH)))
    if not lows:
        return None
    left, right = min(lows), max(highs)
    if right <= left:
        return None
    pad = X_PADDING_FRACTION * (right - left)
    return [left - pad, right + pad]


def plot_cdf(values: Dict[str, np.ndarray], output_dir: str) -> Dict[str, np.ndarray]:
    """Draw one empirical CDF per method; return the finite samples used."""
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    used: Dict[str, np.ndarray] = {}

    for label, column, color, width, zorder in CDF_SERIES:
        raw = values[column]
        finite = np.sort(raw[np.isfinite(raw)])
        used[label] = finite
        if finite.size == 0:
            print(f"[eval_cdf] WARNING: {label} has no finite value, curve skipped.")
            continue
        probabilities = np.arange(1, finite.size + 1) / finite.size
        axis.plot(
            finite,
            probabilities,
            linewidth=width,
            color=color,
            zorder=zorder,
            label=label,
        )

    axis.set_xlabel("Per-location NMSE [dB]", fontsize=LABEL_FONTSIZE)
    axis.set_ylabel("Empirical CDF", fontsize=LABEL_FONTSIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    axis.grid(alpha=0.3, linewidth=0.5)
    axis.legend(fontsize=LEGEND_FONTSIZE, loc="lower right")
    axis.set_ylim(0.0, 1.0)
    axis.set_xlim(-40, 8)

    save_figure(figure, output_dir, FIGURE_STEM)
    return used


# ----------------------------------------------------------------------
# Cross-check
# ----------------------------------------------------------------------
def report_statistics(
    used: Dict[str, np.ndarray], total: int, published: Dict[str, Dict[str, float]]
) -> None:
    """Print recomputed median/mean beside the published table and warn on drift."""
    print("")
    print(f"[eval_cdf] test locations in the CSV : {total}")
    for label, finite in used.items():
        covered = finite.size
        if covered == 0:
            continue
        median = float(np.median(finite))
        mean = float(np.mean(finite))
        entry = published.get(label, {})
        published_mean = entry.get("nmse_mean_dB")
        published_median = entry.get("nmse_median_dB")
        print(
            f"[eval_cdf] {label:<17} n={covered:5d}/{total:<5d} "
            f"median={median:8.3f} dB  mean={mean:8.3f} dB"
        )
        if published_mean is None:
            print(
                f"[eval_cdf]   {'':<15} table   : method not found in "
                f"{TABLE_CSV}, cross-check skipped"
            )
            continue
        median_text = (
            "n/a" if published_median is None else f"{published_median:8.3f} dB"
        )
        print(
            f"[eval_cdf]   {'':<15} table   : "
            f"median={median_text}  mean={published_mean:8.3f} dB"
        )
        delta = abs(mean - published_mean)
        if delta > MEAN_TOLERANCE_DB:
            print(
                f"[eval_cdf]   {'':<15} WARNING : recomputed mean differs from "
                f"{TABLE_CSV} by {delta:.3f} dB (> {MEAN_TOLERANCE_DB} dB)"
            )
        else:
            print(f"[eval_cdf]   {'':<15} OK      : mean matches within {delta:.3f} dB")


def main() -> int:
    print(f"[eval_cdf] per-location : {PER_LOCATION_CSV}")
    print(f"[eval_cdf] table        : {TABLE_CSV}")

    values = read_per_location(PER_LOCATION_CSV)
    published = read_table_means(TABLE_CSV)
    if not published:
        print(f"[eval_cdf] NOTE: {TABLE_CSV} is missing, cross-check skipped.")

    total = int(next(iter(values.values())).size)
    used = plot_cdf(values, OUTPUT_DIR)

    rt_label = "Sionna RT"
    rt_covered = int(used.get(rt_label, np.empty(0)).size)
    print(
        f"[eval_cdf] {rt_label} covers {rt_covered} of {total} test locations "
        f"({100.0 * rt_covered / total:.2f}%); its curve is drawn over that "
        f"subset only."
    )

    report_statistics(used, total, published)

    print("")
    for extension in ("png", "pdf"):
        print(f"[eval_cdf] wrote {os.path.join(OUTPUT_DIR, FIGURE_STEM)}.{extension}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
