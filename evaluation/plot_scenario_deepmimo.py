"""Fig. X(a) -- top view of the DeepMIMO (ASU campus) scene region.

Single panel, no sub-figure label and no caption: the (a)/(b) assembly happens
in LaTeX against ``fig_gap_map_paper.pdf``.

Every UE location of the ASU-campus 16x64 split is drawn as a small gray dot.
The dots sit on a 1 m lattice, so the empty regions *are* the building
footprints -- nothing is drawn for the buildings themselves.

Axis limits, aspect and the BS marker position are pinned to the same values
``evaluation/plot_gap_map_quarter.py`` uses, so the two panels register.

Run with zero arguments::

    python figout/plot_scenario_deepmimo.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(REPO_ROOT, "dataset", "asu_campus_16by64_lt")
OUTPUT_DIR = os.path.join(REPO_ROOT, "figout")

# Pinned to plot_gap_map_quarter.py so panels (a) and (b) share a frame.
X_MIN, X_MAX = 0.0, 185.0
Y_MIN, Y_MAX = 0.0, 160.0
BS_POSITION = (166.0, 104.0)

# The UE lattice pitch is 1 m; at this axes width s=2 pt^2 is roughly a 1 m
# dot, which is what makes the street layout read as continuous ribbons while
# leaving the building footprints empty.
MARKER_SIZE = 2.0
MARKER_ALPHA = 0.5
MARKER_GRAY = "0.45"

AXIS_LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10

# Single-column width. The height only has to leave the equal-aspect axes room
# to use the full width -- ``bbox_inches="tight"`` trims whatever slack the
# constrained layout leaves above and below, so the saved page height is the
# one implied by the 160 m x 185 m data aspect plus the label bands.
FIG_WIDTH = 3.5
FIG_HEIGHT = FIG_WIDTH * (Y_MAX - Y_MIN) / (X_MAX - X_MIN) + 0.6

FIGURE_STEM = "scenario_deepmimo"


def load_positions() -> tuple[np.ndarray, dict]:
    """All UE locations of the split: train + test, in metres."""
    train = sio.loadmat(os.path.join(DATASET_DIR, "train.mat"))["positions"]
    test = sio.loadmat(os.path.join(DATASET_DIR, "test.mat"))["positions"]
    positions = np.vstack([train[:, :2], test[:, :2]]).astype(float)

    # The two splits are disjoint and every row is a distinct lattice cell;
    # assert it rather than trust it, because a duplicate would stack markers.
    keys = np.unique(np.round(positions, 4), axis=0)
    stats = {
        "n_train": int(train.shape[0]),
        "n_test": int(test.shape[0]),
        "n_total": int(positions.shape[0]),
        "n_unique": int(keys.shape[0]),
    }
    if stats["n_unique"] != stats["n_total"]:
        raise SystemExit(
            "[scenario_deepmimo] train and test share a location; refusing to "
            "stack markers"
        )
    if (
        positions[:, 0].min() < X_MIN
        or positions[:, 0].max() > X_MAX
        or positions[:, 1].min() < Y_MIN
        or positions[:, 1].max() > Y_MAX
    ):
        raise SystemExit("[scenario_deepmimo] a location falls outside the axes")
    return positions, stats


def plot(positions: np.ndarray) -> None:
    # No font rcParams: this figure uses matplotlib's default face (DejaVu
    # Sans), which is what the rest of the repo's figures render with.
    figure, axis = plt.subplots(
        figsize=(FIG_WIDTH, FIG_HEIGHT), layout="constrained"
    )
    axis.scatter(
        positions[:, 0],
        positions[:, 1],
        s=MARKER_SIZE,
        c=MARKER_GRAY,
        alpha=MARKER_ALPHA,
        linewidths=0.0,
        rasterized=True,
    )

    axis.set_xlabel("x [m]", fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_ylabel("y [m]", fontsize=AXIS_LABEL_FONTSIZE)
    axis.tick_params(labelsize=TICK_FONTSIZE)
    axis.set_xlim(X_MIN, X_MAX)
    axis.set_ylim(Y_MIN, Y_MAX)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.3, linewidth=0.5)
    axis.set_axisbelow(True)

    axis.plot(
        BS_POSITION[0],
        BS_POSITION[1],
        marker="*",
        color="black",
        markersize=11,
        markeredgewidth=0.0,
        linestyle="none",
        zorder=5,
    )
    axis.annotate(
        "BS",
        xy=BS_POSITION,
        xytext=(6.0, 4.0),
        textcoords="offset points",
        fontsize=AXIS_LABEL_FONTSIZE,
        color="black",
        zorder=5,
    )

    for extension in ("pdf", "png"):
        figure.savefig(
            os.path.join(OUTPUT_DIR, f"{FIGURE_STEM}.{extension}"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(figure)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    positions, stats = load_positions()
    plot(positions)
    print(
        f"  [scenario_deepmimo] {stats['n_total']} UE locations "
        f"({stats['n_train']} train + {stats['n_test']} test), all unique"
    )
    print(f"  [scenario_deepmimo] wrote {OUTPUT_DIR}/{FIGURE_STEM}.pdf and .png")


if __name__ == "__main__":
    main()
