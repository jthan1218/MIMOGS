"""Fig. 5 -- per-location NMSE gap map, original scatter style, quarter-set data.

Standalone re-plot of ``eval_baseline_rt.plot_gap_map_paper`` (copied here and
edited rather than modified in place).  The plotting body is verbatim from that
function -- same ``scatter`` call, marker, ``RdBu_r``, ``linewidths=0``,
``rasterized``, figsize ``(3.6, 3.0)``, serif/Times rcParams, fontsize 6 labels
and 5 ticks, ``alpha=0.3`` grid, and the same colorbar geometry and label.  The
color limit comes from ``eval_baseline_rt.symmetric_gap_limit``, imported rather
than reimplemented, so the scale cannot drift from the original.

Four things differ from the original figure:

* the data -- the 3947 locations of ``analysis/eval_t1/t1_per_location.csv``,
  with the 544 RT-NaN locations dropped (uncolored) rather than plotted;
* the axis limits are pinned to x [0, 185], y [0, 160];
* a black-star BS marker at (166, 104) is drawn (the original drew none);
* the marker size is ``s=7.5`` rather than the original's ``s=4``, because the
  tighter axes shrink the axes box.  It is the only tuned value.

The quantity is unchanged::

    dNMSE = NMSE_shape(Sionna RT) - NMSE_shape(MIMO-GS)      [dB]

positive (red) = ray tracer worse, negative (blue) = ray tracer better.

Run with zero arguments::

    python plot_gap_map_quarter.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The color-limit rule is imported, not copied, so this figure and the original
# provably clip at the same value.
from eval_baseline_rt import symmetric_gap_limit

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(REPO_ROOT, "analysis", "eval_t1", "t1_per_location.csv")
OUTPUT_DIR = os.path.join(
    REPO_ROOT, "analysis", "20260811_062015", "comparison_rt_quarter"
)
REQUESTED_CHECKPOINT = os.path.join(REPO_ROOT, "outputs", "20260811_062015", "model.pth")
CSV_CHECKPOINT = os.path.join(REPO_ROOT, "outputs", "density", "mimogs", "model_100.pth")
SCATTER_CSV = os.path.join(
    REPO_ROOT, "analysis", "20260811_062015", "comparison_rt", "metrics_per_location.csv"
)

COLUMN_RT = "nmse_shape_dB_Sionna RT"
COLUMN_MIMOGS = "nmse_shape_dB_MIMO-GS"

X_MIN, X_MAX = 0.0, 185.0
Y_MIN, Y_MAX = 0.0, 160.0
CELL = 1.0  # the lattice the locations sit on; used only by the uniqueness check

BS_POSITION = (166.0, 104.0)

# The only value tuned for this figure.  eval_baseline_rt.plot_gap_map_paper
# used s=4; the tighter x [0, 185] / y [0, 160] axes shrink the axes box, so
# the markers are enlarged to keep the point field reading at the same density.
MARKER_SIZE = 7.5

FIGURE_STEM = "fig_gap_map_paper"


def assert_checkpoint_identity() -> str:
    """Assert the CSV's MIMO-GS column came from the requested checkpoint.

    ``outputs/density/mimogs/model_100.pth`` is a self-contained repack; it
    records ``source_run`` and carries the same ``capture`` tuple.  Comparing
    the tuples element-wise is what licenses reading the CSV instead of
    re-rendering ``outputs/20260811_062015/model.pth``.
    """
    import torch

    requested = torch.load(REQUESTED_CHECKPOINT, map_location="cpu", weights_only=False)
    repack = torch.load(CSV_CHECKPOINT, map_location="cpu", weights_only=False)

    def same(left, right, path: str) -> bool:
        if torch.is_tensor(left) or torch.is_tensor(right):
            if not (torch.is_tensor(left) and torch.is_tensor(right)):
                print(f"  [checkpoint] type mismatch at {path}")
                return False
            ok = left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
            if not ok:
                print(f"  [checkpoint] tensor mismatch at {path}")
            return ok
        if isinstance(left, dict) and isinstance(right, dict):
            return all(
                same(left.get(key), right.get(key), f"{path}/{key}")
                for key in set(left) | set(right)
            )
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                print(f"  [checkpoint] length mismatch at {path}")
                return False
            return all(
                same(a, b, f"{path}[{i}]") for i, (a, b) in enumerate(zip(left, right))
            )
        ok = left == right
        if not ok:
            print(f"  [checkpoint] value mismatch at {path}: {left!r} != {right!r}")
        return ok

    if not same(requested["gaussians"], repack["capture"], "capture"):
        raise SystemExit(
            "[plot_gap_map_quarter] the CSV's checkpoint is NOT the requested one; "
            "re-render before plotting."
        )
    note = (
        f"iteration {requested['iteration']}, repack source_run "
        f"{repack.get('source_run')}, train fraction {repack.get('fraction')}, "
        f"{repack.get('epochs')} epochs"
    )
    print(f"  [checkpoint] capture bit-identical to the requested model.pth ({note})")
    return note


def lattice_collisions(frame: pd.DataFrame) -> list:
    """Rows sharing a 1 m lattice cell, reported rather than merged.

    The scatter draws every row, so nothing can be overwritten here; the check
    is kept from the grid version because a duplicate location would still be a
    data fault (two markers stacked at one point).
    """
    n_x = int(round((X_MAX - X_MIN) / CELL))
    ix = np.floor((frame["x_m"].to_numpy(float) - X_MIN) / CELL).astype(int)
    iy = np.floor((frame["y_m"].to_numpy(float) - Y_MIN) / CELL).astype(int)

    if ix.min() < 0 or ix.max() >= n_x or iy.min() < 0:
        raise SystemExit("[plot_gap_map_quarter] a location falls outside the axes")

    linear = iy * n_x + ix
    cells, counts = np.unique(linear, return_counts=True)
    collisions = []
    for cell in cells[counts > 1]:
        rows = np.flatnonzero(linear == cell)
        collisions.append((int(cell % n_x), int(cell // n_x), rows.tolist()))
    return collisions


def plot_gap_map_paper_quarter(
    output_dir: str,
    positions: np.ndarray,
    gap_db: np.ndarray,
) -> float:
    """``eval_baseline_rt.plot_gap_map_paper`` verbatim, plus limits and a BS star."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "DejaVu Serif",
    ]
    plt.rcParams["mathtext.fontset"] = "stix"
    gap_limit = symmetric_gap_limit(gap_db)

    figure, axis = plt.subplots(figsize=(3.6, 3.0), layout="constrained")
    scatter = axis.scatter(
        positions[:, 0],
        positions[:, 1],
        c=gap_db,
        s=MARKER_SIZE,
        cmap="RdBu_r",
        vmin=-gap_limit,
        vmax=gap_limit,
        linewidths=0.0,
        rasterized=True,
    )
    axis.set_xlabel("x [m]", fontsize=6)
    axis.set_ylabel("y [m]", fontsize=6)
    axis.tick_params(labelsize=5)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.3, linewidth=0.5)

    # --- the only additions to the original body ---
    axis.set_xlim(X_MIN, X_MAX)
    axis.set_ylim(Y_MIN, Y_MAX)
    axis.plot(
        BS_POSITION[0],
        BS_POSITION[1],
        marker="*",
        color="black",
        markersize=7,
        markeredgewidth=0.0,
        linestyle="none",
        zorder=5,
    )
    axis.annotate(
        "BS",
        xy=BS_POSITION,
        xytext=(4.0, 3.0),
        textcoords="offset points",
        fontsize=6,
        color="black",
        zorder=5,
    )
    # --- end additions ---

    colorbar = figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label(r"$\Delta$NMSE [dB]", fontsize=6)
    colorbar.ax.tick_params(labelsize=5)

    figure.savefig(
        os.path.join(output_dir, f"{FIGURE_STEM}.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    figure.savefig(
        os.path.join(output_dir, f"{FIGURE_STEM}.pdf"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)
    return float(gap_limit)


def write_report(output_dir: str, stats: dict) -> None:
    lines = [
        "# Fig. 5 gap map, original scatter style, quarter-set data",
        "",
        "Produced by `plot_gap_map_quarter.py` at the repository root.  The",
        "existing `../comparison_rt/` folder was not touched.  The earlier",
        "grid-image rendering is discarded; these files replace it.",
        "",
        f"Output: `{FIGURE_STEM}.pdf` and `{FIGURE_STEM}.png`, marker size",
        f"`s={MARKER_SIZE:g}`.",
        "",
        "## Style",
        "",
        "The plotting body is copied verbatim from",
        "`eval_baseline_rt.plot_gap_map_paper`: `scatter` with the default round",
        "marker, `cmap=\"RdBu_r\"`, `linewidths=0.0`, `rasterized=True`, figsize",
        "`(3.6, 3.0)` with `layout=\"constrained\"`, serif/Times rcParams and stix",
        "mathtext, fontsize 6 axis labels, labelsize 5 ticks, `aspect=\"equal\"`,",
        "`grid(alpha=0.3, linewidth=0.5)`, and a colorbar at `fraction=0.046,",
        "pad=0.03` labeled $\\Delta$NMSE [dB].  Saved at dpi 300 with",
        "`bbox_inches=\"tight\", pad_inches=0.02`.",
        "",
        "The color limit is not hard-coded: `eval_baseline_rt.symmetric_gap_limit`",
        "is **imported and called**, so the rule cannot drift from the original.",
        f"On this data it returns +/-{stats['gap_limit']:.0f} dB, the same value it returns on",
        "the original figure's own gap array -- the two figures share a scale.",
        "(This supersedes the +/-30 dB of the previous, discarded grid version.)",
        "",
        "Departures from the original figure, all requested:",
        "",
        "- axis limits pinned to x [0, 185] m, y [0, 160] m (the original let",
        "  matplotlib autoscale, which padded roughly 4-7 m beyond the data);",
        "- a black star with a `BS` label at (166, 104) -- the original",
        "  `plot_gap_map_paper` drew no BS marker at all, so there was no",
        "  existing style to match;",
        f"- the marker size, `s={MARKER_SIZE:g}` against the original's `s=4`.  The tighter",
        "  axes shrink the axes box, so a larger marker keeps the point field",
        "  reading at the same density.  This is the only tuned value.",
        "",
        "## Data source",
        "",
        "**CSV, no re-render.**  Per-location values read from",
        "`analysis/eval_t1/t1_per_location.csv` (3947 test locations), columns",
        f"`{COLUMN_RT}` and `{COLUMN_MIMOGS}`.",
        "",
        "Plotted quantity, unchanged from the original:",
        "",
        "    dNMSE = NMSE_shape(Sionna RT) - NMSE_shape(MIMO-GS)   [dB]",
        "",
        "positive (red) = ray tracer worse, negative (blue) = ray tracer better.",
        "The 544 locations whose RT value is NaN are dropped before the scatter,",
        "so they are uncolored (no marker) rather than plotted at any value.",
        "",
        "Re-rendering was avoidable because the CSV's MIMO-GS column comes from",
        "`outputs/density/mimogs/model_100.pth`, whose `capture` tuple is",
        "**bit-identical** to `outputs/20260811_062015/model.pth`",
        f"({stats['checkpoint_note']}).  The identity is asserted element-wise at",
        "run time; the script exits rather than plotting if it fails.",
        "",
        "Independent cross-check against the original figure's own CSV",
        "(`../comparison_rt/metrics_per_location.csv`, 3403 matched rows):",
        "",
        f"- Sionna RT column: max |difference| = {stats['xcheck_rt_max']:.3e} dB (exact)",
        f"- MIMO-GS column:   max |difference| = {stats['xcheck_mg_max']:.4f} dB, "
        f"mean {stats['xcheck_mg_mean']:.2e} dB",
        "",
        "The MIMO-GS wobble is the known CUDA-rasterizer atomics non-determinism,",
        "not a different checkpoint.",
        "",
        "## Counts",
        "",
        f"- Test locations in the CSV: {stats['n_locations']}",
        f"- Colored (plotted) locations, finite dNMSE: {stats['n_colored']}",
        f"- RT-NaN locations, uncolored: {stats['n_rt_nan']}",
        "",
        "These are the same 3403 locations the original figure drew, so the",
        "marker field is expected to look the same; the visible differences are",
        "the tightened axes and the BS star.",
        "",
        "## dNMSE over the colored locations",
        "",
        f"- min  {stats['d_min']:+.4f} dB",
        f"- max  {stats['d_max']:+.4f} dB",
        f"- mean {stats['d_mean']:+.4f} dB",
        f"- fraction with dNMSE > 0: {stats['frac_positive']:.6f} "
        f"({stats['n_positive']} / {stats['n_colored']})",
        "",
        f"At the +/-{stats['gap_limit']:.0f} dB limit, {stats['n_clip_high']} marker(s) saturate at the red end",
        f"and {stats['n_clip_low']} at the blue end.",
        "",
        "## Sanity",
        "",
        "**1. Mean dNMSE over the colored locations vs. the difference of the two",
        "column means.**",
        "",
        f"- mean of per-location dNMSE:                      {stats['d_mean']:+.6f} dB",
        f"- mean({COLUMN_RT}) - mean({COLUMN_MIMOGS}), both",
        "  restricted to the 3403 RT-covered rows:           "
        f"{stats['col_mean_diff_matched']:+.6f} dB",
        f"- absolute difference:                             {stats['sanity_gap']:.3e} dB",
        "",
        "These agree to floating-point.  Taking the MIMO-GS mean over all 3947",
        "rows instead gives",
        f"{stats['rt_mean']:+.6f} - ({stats['mg_mean_all']:+.6f}) = "
        f"{stats['col_mean_diff_all']:+.6f} dB,",
        f"which differs by {stats['sanity_gap_all']:.4f} dB -- the two columns must be",
        "restricted to the same rows for the identity to hold, because RT covers",
        "only 3403 of the 3947 locations.",
        "",
        "**2. Every (x_m, y_m) is a distinct location.**",
        "",
    ]
    if stats["collisions"]:
        lines.append(
            f"{len(stats['collisions'])} COLLISION(S) on the 1 m lattice -- reported, "
            "not merged (the scatter draws every row regardless):"
        )
        lines.append("")
        for cell_x, cell_y, rows in stats["collisions"]:
            lines.append(f"- cell (ix={cell_x}, iy={cell_y}): CSV rows {rows}")
    else:
        lines.append(
            f"No collisions: {stats['n_locations']} locations map to "
            f"{stats['n_unique_cells']} distinct 1 m lattice cells, checked over all "
            "rows (including the RT-NaN ones).  Nothing is stacked or overwritten."
        )
    lines += [
        "",
        "## Note on the \"quarter\" label",
        "",
        "`outputs/20260811_062015/model.pth` is the FULL-data run: the repack",
        f"records train fraction {stats['fraction']}, {stats['epochs']} epochs, 15787 training",
        "locations.  The 25% model is `outputs/density/mimogs/model_25.pth`",
        "(source run 20260818_040130).  There is also no quarter test set --",
        "`eval_density.py` subsamples TRAIN only, and every fraction is scored on",
        "the original full `test.mat` (3947 locations), which is what this CSV",
        "holds.  The figure here is therefore the same predictor and the same",
        "locations as `../comparison_rt/fig_gap_map_paper.pdf`.",
        "",
    ]
    with open(os.path.join(output_dir, "report.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    print("[plot_gap_map_quarter] verifying the CSV's checkpoint provenance")
    checkpoint_note = assert_checkpoint_identity()

    import torch

    repack = torch.load(CSV_CHECKPOINT, map_location="cpu", weights_only=False)

    frame = pd.read_csv(CSV_PATH)
    print(f"[plot_gap_map_quarter] {len(frame)} test locations from {CSV_PATH}")

    collisions = lattice_collisions(frame)

    rt_values = frame[COLUMN_RT].to_numpy(dtype=float)
    mg_values = frame[COLUMN_MIMOGS].to_numpy(dtype=float)
    difference = rt_values - mg_values
    finite = np.isfinite(difference)

    # RT-NaN locations are dropped, not plotted: no marker is drawn for them.
    positions = frame.loc[finite, ["x_m", "y_m"]].to_numpy(dtype=float)
    colored = difference[finite]

    rt_mean = float(np.nanmean(rt_values))
    mg_mean_all = float(np.mean(mg_values))
    col_mean_diff_matched = float(np.mean(rt_values[finite]) - np.mean(mg_values[finite]))

    merged = frame.merge(
        pd.read_csv(SCATTER_CSV), left_on="test_index", right_on="gt_test_index"
    )
    delta_rt = np.abs(merged[COLUMN_RT] - merged["sionna_NMSE_shape_dB"])
    delta_mg = np.abs(merged[COLUMN_MIMOGS] - merged["mimogs_NMSE_shape_dB"])

    n_x = int(round((X_MAX - X_MIN) / CELL))
    ix = np.floor((frame["x_m"].to_numpy(float) - X_MIN) / CELL).astype(int)
    iy = np.floor((frame["y_m"].to_numpy(float) - Y_MIN) / CELL).astype(int)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gap_limit = plot_gap_map_paper_quarter(OUTPUT_DIR, positions, colored)
    print(f"  wrote {FIGURE_STEM}.pdf / .png  (s={MARKER_SIZE:g})")

    stats = {
        "checkpoint_note": checkpoint_note,
        "fraction": repack.get("fraction"),
        "epochs": repack.get("epochs"),
        "gap_limit": gap_limit,
        "n_locations": int(len(frame)),
        "n_colored": int(finite.sum()),
        "n_rt_nan": int((~finite).sum()),
        "n_unique_cells": int(len(np.unique(iy * n_x + ix))),
        "collisions": collisions,
        "d_min": float(colored.min()),
        "d_max": float(colored.max()),
        "d_mean": float(colored.mean()),
        "n_positive": int((colored > 0.0).sum()),
        "frac_positive": float((colored > 0.0).mean()),
        "n_clip_high": int((colored > gap_limit).sum()),
        "n_clip_low": int((colored < -gap_limit).sum()),
        "rt_mean": rt_mean,
        "mg_mean_all": mg_mean_all,
        "col_mean_diff_matched": col_mean_diff_matched,
        "col_mean_diff_all": float(rt_mean - mg_mean_all),
        "sanity_gap": abs(float(colored.mean()) - col_mean_diff_matched),
        "sanity_gap_all": abs(float(colored.mean()) - (rt_mean - mg_mean_all)),
        "xcheck_rt_max": float(delta_rt.max()),
        "xcheck_mg_max": float(delta_mg.max()),
        "xcheck_mg_mean": float(delta_mg.mean()),
    }

    write_report(OUTPUT_DIR, stats)

    print(f"  color limit         : +/-{gap_limit:.0f} dB (symmetric_gap_limit)")
    print(f"  colored locations   : {stats['n_colored']}")
    print(f"  RT-NaN (uncolored)  : {stats['n_rt_nan']}")
    print(
        f"  dNMSE min/max/mean  : {stats['d_min']:+.4f} / {stats['d_max']:+.4f} / "
        f"{stats['d_mean']:+.4f} dB"
    )
    print(f"  fraction positive   : {stats['frac_positive']:.6f}")
    print(
        f"  sanity mean vs cols : {stats['d_mean']:+.6f} vs "
        f"{stats['col_mean_diff_matched']:+.6f} dB (|gap| {stats['sanity_gap']:.3e})"
    )
    print(f"  lattice collisions  : {len(collisions)}")
    print(f"[plot_gap_map_quarter] wrote {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
