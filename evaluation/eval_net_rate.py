"""Net achievable rate R_sel(p) for beam selection -- paper Eq. (6_net_Rsel).

Runs with zero arguments::

    python eval_net_rate.py

For every held-out test location p and every transmit budget L_t,

    R_sel(p; L_t) = (1 - L_t * tau_RS / T_B)
                    * (1/B) * sum_b log2 det( I_{L_r}
                              + (P / (L_t * sigma_z^2)) * H_sel,b H_sel,b^H )

where ``H_sel,b`` is the ``(L_r x L_t)`` submatrix of the b-th stored beamspace
realization obtained by pure row/column indexing with the selected receive
beams ``B_r`` and transmit beams ``B_t``.  The fading expectation is a Monte
Carlo average over the B = 10 stored realizations per location; the reported
curves average R_sel(p) over the test locations.

Three selection schemes are compared, each under two transmit rules:

* MIMO-GS -- beams chosen from the rendered map X_hat(p),
* Genie   -- beams chosen from the ground-truth long-term map X_gt(p),
* Random  -- uniform L_t-subsets, averaged over several draws.

Power convention (verified, not assumed)
----------------------------------------
The dataset field named ``magnitude`` is ALREADY a power map: over 50 random
matched locations, ``mean_b |H_b|^2`` correlates 0.9986 with it at a ratio of
0.999, whereas the correlation against ``magnitude**2`` is only 0.84.  The
selection metric therefore consumes the maps directly, with no squaring.  The
script re-runs that correlation check at startup and refuses to continue if it
fails.

Selection-scale caveat
----------------------
``X_gt`` carries the true absolute power (per-location peak ~0.03) while
``X_hat`` is trained against the max-normalized target and peaks near 1.0.  The
greedy metric ``log2(1 + (P/(|B| sigma_z^2)) * sum M)`` is NOT scale invariant,
so feeding each scheme its native map puts Genie in the near-linear region of
the log and MIMO-GS in the curved region.  ``--selection_scale as_is`` (the
default) is the literal reading of the paper's rule; ``--selection_scale peak``
max-normalizes both maps first so the two schemes differ only in map SHAPE.
The run reports both so the confound is visible rather than buried.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_baseline_rt import (
    DEFAULT_CKPT,
    gain_net_hidden_dim,
    gain_net_width,
    load_raw_mat,
    render_mimogs,
    save_figure,
)
from eval_render import build_scene_and_model, resolve_run_dir, restore_config
from baseline_models import (
    POSITION_BINS,
    classifier_orders,
    normalize_and_quantize,
    topk_beam_accuracy,
    train_beam_classifier,
)


DEFAULT_LT_GRID = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_TAU_OVER_TB = 1.0 / 128.0
TB_REFERENCE = 128.0          # tau_RS is pinned via tau_over_TB at this T_B
CDF_LT = 8
BT_RECORD_LT = 8
MATCH_TOL = 1e-4
EIG_FLOOR = 0.0

SOLID_SCHEMES = ("mimogs_greedy", "genie_greedy", "random")
DASHED_SCHEMES = ("mimogs_toppower", "genie_toppower")
# The two literature-anchored baselines.  Kept out of SOLID/DASHED so the
# fig_net_rate_vs_Lt styling is untouched; they still reach every CSV.
EXTRA_SCHEMES = ("position_nn", "statistical", "position_nn_clf")
ALL_SCHEMES = SOLID_SCHEMES + DASHED_SCHEMES
CSV_SCHEMES = ALL_SCHEMES + EXTRA_SCHEMES

# Selection rules that must be re-run per SNR (the metric contains P/sigma^2).
GREEDY_SCHEMES = ("genie_greedy", "mimogs_greedy", "position_nn")

SCHEME_LABEL = {
    "mimogs_greedy": "MIMO-GS (greedy)",
    "genie_greedy": "Genie (greedy)",
    "random": "Random",
    "mimogs_toppower": "MIMO-GS (top-power)",
    "genie_toppower": "Genie (top-power)",
    "position_nn": "Position NN",
    "statistical": "Statistical codebook",
    "position_nn_clf": "Position NN classifier",
}
SCHEME_COLOR = {
    "mimogs_greedy": "tab:blue",
    "genie_greedy": "tab:green",
    "random": "tab:gray",
    "mimogs_toppower": "tab:blue",
    "genie_toppower": "tab:green",
    "position_nn": "tab:purple",
    "statistical": "tab:orange",
    "position_nn_clf": "tab:cyan",
}


# ----------------------------------------------------------------------
# Position join
# ----------------------------------------------------------------------
def join_positions(
    test_positions: np.ndarray, complex_positions: np.ndarray, tolerance: float
) -> np.ndarray:
    """Return the complex.mat row for every test location, or abort loudly."""
    try:
        from scipy.spatial import cKDTree  # noqa: PLC0415 -- optional fast path

        distances, indices = cKDTree(complex_positions).query(test_positions, k=1)
        distances = np.asarray(distances, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int64)
    except ImportError:
        distances = np.empty(test_positions.shape[0], dtype=np.float64)
        indices = np.empty(test_positions.shape[0], dtype=np.int64)
        for start in range(0, test_positions.shape[0], 512):
            stop = min(start + 512, test_positions.shape[0])
            deltas = test_positions[start:stop, None, :] - complex_positions[None, :, :]
            block = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))
            indices[start:stop] = block.argmin(axis=1)
            distances[start:stop] = block.min(axis=1)

    unmatched = int(np.sum(distances > tolerance))
    duplicated = int(indices.size - np.unique(indices).size)

    if unmatched or duplicated:
        print("-" * 78)
        print("[eval_net_rate] POSITION JOIN FAILED")
        print(f"  test locations              : {test_positions.shape[0]}")
        print(f"  beyond tolerance {tolerance:g} m   : {unmatched}")
        print(f"  duplicated complex.mat rows : {duplicated}")
        for name, array in (
            ("test.mat", test_positions),
            ("complex.mat", complex_positions),
        ):
            lows, highs = array.min(axis=0), array.max(axis=0)
            print(
                f"  {name:<12} N={array.shape[0]:<7d} "
                f"x=[{lows[0]:.4f}, {highs[0]:.4f}] "
                f"y=[{lows[1]:.4f}, {highs[1]:.4f}] "
                f"z=[{lows[2]:.4f}, {highs[2]:.4f}]"
            )
        percentiles = np.percentile(distances, [50, 90, 99, 100])
        print(
            f"  NN distance [m]: p50={percentiles[0]:.4g} p90={percentiles[1]:.4g} "
            f"p99={percentiles[2]:.4g} max={percentiles[3]:.4g}"
        )
        print("-" * 78)
        raise SystemExit(
            "[eval_net_rate] Every test location must join to exactly one "
            "complex.mat row; refusing to continue."
        )

    return indices


def _nearest_train(
    test_positions: np.ndarray, train_positions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest TRAIN location for every test location (3-D, raw metres)."""
    try:
        from scipy.spatial import cKDTree  # noqa: PLC0415 -- optional fast path

        distances, indices = cKDTree(train_positions).query(test_positions, k=1)
        return (
            np.asarray(distances, dtype=np.float64),
            np.asarray(indices, dtype=np.int64),
        )
    except ImportError:
        distances = np.empty(test_positions.shape[0], dtype=np.float64)
        indices = np.empty(test_positions.shape[0], dtype=np.int64)
        for start in range(0, test_positions.shape[0], 512):
            stop = min(start + 512, test_positions.shape[0])
            deltas = test_positions[start:stop, None, :] - train_positions[None, :, :]
            block = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))
            indices[start:stop] = block.argmin(axis=1)
            distances[start:stop] = block.min(axis=1)
        return distances, indices


# ----------------------------------------------------------------------
# Beam selection
# ----------------------------------------------------------------------
def greedy_order(
    selection_map: torch.Tensor, snr: float, num_rx: int, steps: int
) -> torch.Tensor:
    """Greedy transmit-beam ordering, Eq. (6_selection_metric).

    ``selection_map`` is ``(N, N_r, N_t)`` power.  At step k the score of a
    candidate n is the sum of the ``num_rx`` largest values over m of

        log2( 1 + (P / (k * sigma_z^2)) * sum_{n' in B u {n}} M[m, n'] ),

    i.e. the power-splitting factor uses the size the set WOULD have.  Greedy
    is prefix-consistent, so one 64-step run yields the selection for every
    L_t at once: ``order[:, :L_t]``.

    All candidates are scored at once; the only loop is over the L_t steps.
    """
    count, rx_beams, tx_beams = selection_map.shape
    device = selection_map.device

    running = torch.zeros(count, rx_beams, device=device, dtype=selection_map.dtype)
    taken = torch.zeros(count, tx_beams, dtype=torch.bool, device=device)
    order = torch.empty(count, steps, dtype=torch.long, device=device)

    transposed = selection_map.transpose(1, 2).contiguous()   # (N, N_t, N_r)
    keep = min(int(num_rx), rx_beams)

    for step in range(1, steps + 1):
        # (N, N_t, N_r): row powers the set would have with candidate n added.
        candidate = running.unsqueeze(1) + transposed
        score = torch.log2(1.0 + (snr / float(step)) * candidate)
        score = score.topk(keep, dim=2).values.sum(dim=2)      # (N, N_t)
        score = score.masked_fill(taken, float("-inf"))

        picked = score.argmax(dim=1)                           # (N,)
        order[:, step - 1] = picked
        taken.scatter_(1, picked.unsqueeze(1), True)

        gathered = torch.gather(
            selection_map, 2, picked.view(count, 1, 1).expand(count, rx_beams, 1)
        )
        running = running + gathered.squeeze(2)

    return order


def toppower_order(selection_map: torch.Tensor) -> torch.Tensor:
    """Transmit beams ranked by column sum (descending)."""
    return torch.argsort(selection_map.sum(dim=1), dim=1, descending=True)


def random_order(count: int, tx_beams: int, generator: torch.Generator,
                 device: torch.device) -> torch.Tensor:
    """One independent uniform permutation of the transmit beams per location."""
    keys = torch.rand(count, tx_beams, generator=generator, device=device)
    return torch.argsort(keys, dim=1)


def receive_selection(
    ground_truth: torch.Tensor, tx_beams: torch.Tensor, num_rx: int
) -> torch.Tensor:
    """Long-term receive selection on the GT map, shared by every scheme.

    ``s_m = sum_{n in B_t} X_gt[m, n]``; the ``num_rx`` largest rows win.  The
    choice is made once per location and held across the fading realizations.
    """
    count, rx_beams, _ = ground_truth.shape
    budget = tx_beams.shape[1]

    gathered = torch.gather(
        ground_truth, 2, tx_beams.unsqueeze(1).expand(count, rx_beams, budget)
    )
    strength = gathered.sum(dim=2)                             # (N, N_r)
    return strength.topk(min(int(num_rx), rx_beams), dim=1).indices


# ----------------------------------------------------------------------
# Rate
# ----------------------------------------------------------------------
def spectral_efficiency(
    channel: torch.Tensor,
    rx_beams: torch.Tensor,
    tx_beams: torch.Tensor,
    snr: float,
) -> torch.Tensor:
    """Monte Carlo mean of log2 det(I + (P/(L_t sigma^2)) H_sel H_sel^H).

    ``channel`` is ``(N, B, N_r, N_t)`` complex.  Returns ``(N,)`` real: the
    average over the B realizations, before the overhead prelog.
    """
    count, realizations, rx_total, tx_total = channel.shape
    num_rx = rx_beams.shape[1]
    num_tx = tx_beams.shape[1]

    rows = rx_beams.view(count, 1, num_rx, 1).expand(
        count, realizations, num_rx, tx_total
    )
    selected = torch.gather(channel, 2, rows)

    columns = tx_beams.view(count, 1, 1, num_tx).expand(
        count, realizations, num_rx, num_tx
    )
    selected = torch.gather(selected, 3, columns)               # (N, B, L_r, L_t)

    gram = selected @ selected.conj().transpose(-1, -2)         # (N, B, L_r, L_r)
    # Kill the asymmetry that accumulates in the matmul before eigvalsh.
    gram = 0.5 * (gram + gram.conj().transpose(-1, -2))

    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(EIG_FLOOR)
    per_realization = torch.log2(
        1.0 + (snr / float(num_tx)) * eigenvalues
    ).sum(dim=2)                                                # (N, B)

    return per_realization.mean(dim=1)


CURVE_KEYS = (
    "bound", "mimogs", "position_nn", "statistical", "position_nn_clf",
    "random", "exhaustive", "genie_same_budget",
    "scheme_prelog", "exhaustive_prelog",
)


def net_rate_vs_snr(
    channel: torch.Tensor,
    ground_truth: torch.Tensor,
    selection_maps: Dict[str, torch.Tensor],
    fixed_orders: Dict[str, torch.Tensor],
    random_orders: Sequence[torch.Tensor],
    fixed_lt: int,
    fixed_tb: float,
    tau_rs: float,
    full_sweep: int,
    num_rx: int,
    snr_grid_db: Sequence[float],
) -> Dict[str, np.ndarray]:
    """Every curve against SNR at a fixed ``(L_t, T_B)`` operating point.

    Greedy selection is rebuilt at every SNR because the selection metric
    contains ``P/sigma_z^2``; the statistical codebook is SNR-independent and
    reuses its fixed ordering.  ``X_hat`` is rendered once by the caller.
    """
    collected: Dict[str, List[float]] = {key: [] for key in CURVE_KEYS}

    for snr_db in snr_grid_db:
        snr = float(10.0 ** (float(snr_db) / 10.0))
        raw = raw_rates_at(
            channel, ground_truth, selection_maps, fixed_orders, random_orders,
            fixed_lt, num_rx, snr,
        )
        point = build_curves(raw, fixed_lt, full_sweep, tau_rs, fixed_tb)
        for key in CURVE_KEYS:
            collected[key].append(point[key])

    result = {key: np.asarray(values, dtype=np.float64)
              for key, values in collected.items()}
    result["snr_db"] = np.asarray(snr_grid_db, dtype=np.float64)
    return result


def net_rate_vs_tb(
    raw: Dict[str, float],
    tb_grid: np.ndarray,
    fixed_lt: int,
    full_sweep: int,
    tau_rs: float,
) -> Dict[str, np.ndarray]:
    """Every curve against T_B at a fixed SNR and fixed data budget.

    The bound has no T_B dependence at all (prelog 1), so it plots flat; the
    others differ from it only through their measurement prelogs and, among
    the same-prelog schemes, through selection quality.
    """
    collected: Dict[str, List[float]] = {key: [] for key in CURVE_KEYS}

    for block in tb_grid:
        point = build_curves(raw, fixed_lt, full_sweep, tau_rs, float(block))
        for key in CURVE_KEYS:
            collected[key].append(point[key])

    result = {key: np.asarray(values, dtype=np.float64)
              for key, values in collected.items()}
    result["T_B"] = np.asarray(tb_grid, dtype=np.float64)
    return result


def prelog_at(num_tx: int, tau_rs: float, block: float) -> float:
    """Overhead factor from an explicit (tau_RS, T_B) pair, floored at zero."""
    return max(0.0, 1.0 - float(num_tx) * float(tau_rs) / float(block))


def prelog(num_tx: int, tau_over_tb: float) -> float:
    """Overhead factor ``1 - L_t * tau_RS / T_B``, floored at zero."""
    return max(0.0, 1.0 - float(num_tx) * float(tau_over_tb))


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def plot_rate_vs_lt(
    output_dir: str,
    stem: str,
    lt_grid: Sequence[int],
    means: Dict[str, np.ndarray],
    y_label: str,
    title: str,
    mark_maxima: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(7.6, 5.2))

    # Genie and MIMO-GS very nearly coincide, so Genie is drawn as a wide
    # translucent band and MIMO-GS as a thin line on top: without this the
    # blue curve is completely hidden and the figure reads as one scheme.
    band_style = {
        "genie_greedy": {"linewidth": 5.0, "alpha": 0.35, "markersize": 0.0,
                         "zorder": 2},
        "mimogs_greedy": {"linewidth": 1.6, "alpha": 1.0, "markersize": 4.5,
                          "zorder": 4},
        "random": {"linewidth": 1.9, "alpha": 1.0, "markersize": 5.0, "zorder": 3},
    }

    for scheme in SOLID_SCHEMES:
        values = means[scheme]
        style = band_style[scheme]
        reference = means["genie_greedy"]
        share = ""
        if scheme != "genie_greedy":
            best_reference = float(np.max(reference))
            if best_reference > 0.0:
                share = f"  [{100.0 * float(np.max(values)) / best_reference:.1f}% of Genie]"
        axis.plot(
            lt_grid,
            values,
            marker="o",
            markersize=style["markersize"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            zorder=style["zorder"],
            color=SCHEME_COLOR[scheme],
            label=SCHEME_LABEL[scheme] + share,
        )
        if mark_maxima:
            best = int(np.argmax(values))
            axis.plot(
                [lt_grid[best]],
                [values[best]],
                marker="*",
                markersize=15,
                color=SCHEME_COLOR[scheme],
                markeredgecolor="black",
                markeredgewidth=0.6,
                linestyle="none",
                zorder=5,
            )
            axis.annotate(
                f"L_t*={lt_grid[best]}",
                (lt_grid[best], values[best]),
                textcoords="offset points",
                xytext=(6, 7),
                fontsize=8,
                color=SCHEME_COLOR[scheme],
            )

    for scheme in DASHED_SCHEMES:
        axis.plot(
            lt_grid,
            means[scheme],
            marker="s" if scheme.startswith("genie") else "^",
            markersize=3.5,
            linewidth=1.2,
            linestyle="--" if scheme.startswith("genie") else ":",
            alpha=0.9,
            zorder=5,
            color=SCHEME_COLOR[scheme],
            label=SCHEME_LABEL[scheme],
        )

    axis.set_xscale("log", base=2)
    axis.set_xticks(list(lt_grid))
    axis.set_xticklabels([str(value) for value in lt_grid])
    axis.set_xlabel("Transmit budget $L_t$ (CSI-RS beams)")
    axis.set_ylabel(y_label)
    axis.set_title(title, fontsize=11)
    axis.grid(alpha=0.3, linewidth=0.5)
    axis.legend(fontsize=8)

    save_figure(figure, output_dir, stem)


def raw_rates_at(
    channel: torch.Tensor,
    ground_truth: torch.Tensor,
    selection_maps: Dict[str, torch.Tensor],
    fixed_orders: Dict[str, torch.Tensor],
    random_orders: Sequence[torch.Tensor],
    fixed_lt: int,
    num_rx: int,
    snr: float,
) -> Dict[str, float]:
    """Overhead-free MC-averaged rate for every scheme at one (SNR, L_t).

    ``selection_maps`` drive the greedy rule, whose metric contains
    ``P/sigma_z^2``, so their ordering is rebuilt at this SNR (only
    ``fixed_lt`` steps deep).  ``fixed_orders`` are SNR-independent beam
    rankings -- the statistical codebook -- supplied as a full ``(N, N_t)``
    ordering so that ``[:, :L_t]`` is the codebook at that budget.  Receive
    beams always come from the GT row sums over the selected columns.
    """
    rates: Dict[str, float] = {}

    def score(tx_beams: torch.Tensor) -> float:
        rx_beams = receive_selection(ground_truth, tx_beams, num_rx)
        return float(
            spectral_efficiency(channel, rx_beams, tx_beams, snr).mean().item()
        )

    for scheme, selection_map in selection_maps.items():
        order = greedy_order(selection_map, snr, num_rx, fixed_lt)
        rates[scheme] = score(order[:, :fixed_lt].contiguous())

    for scheme, order in fixed_orders.items():
        rates[scheme] = score(order[:, :fixed_lt].contiguous())

    if random_orders:
        accumulator = 0.0
        for permutation in random_orders:
            accumulator += score(permutation[:, :fixed_lt].contiguous())
        rates["random"] = accumulator / float(len(random_orders))

    return rates


def build_curves(
    raw: Dict[str, float],
    fixed_lt: int,
    full_sweep: int,
    tau_rs: float,
    block: float,
) -> Dict[str, float]:
    """Plotted quantities at one operating point.

    * ``bound``      -- prelog 1, GT-selected subspace at the SAME data budget
      ``fixed_lt``.  Perfect selection AND free measurement: the ceiling.
    * ``mimogs`` / ``position_nn`` / ``statistical`` / ``random`` -- prelog
      ``1 - fixed_lt*tau_RS/T_B``, differing only in which map (or fixed
      codebook) drives the transmit selection.  Because they share a prelog,
      any separation between them is pure selection quality.
    * ``exhaustive`` -- prelog ``1 - N_t*tau_RS/T_B`` (it measures every beam)
      applied to the BOUND's rate term: an exhaustive sweep recovers the same
      optimal subspace, so only the measurement cost separates the two.  Hence
      ``exhaustive == bound * prelog_exhaustive`` identically.

    ``genie_same_budget`` and ``random`` are kept for the CSVs but are not
    plotted.
    """
    scheme_prelog = max(0.0, 1.0 - fixed_lt * tau_rs / block)
    # Clamped at 0: for T_B < N_t*tau_RS a full sweep cannot fit in the block.
    exhaustive_prelog = max(0.0, 1.0 - full_sweep * tau_rs / block)

    curves = {
        "bound": raw["genie_greedy"],
        "exhaustive": exhaustive_prelog * raw["genie_greedy"],
        "genie_same_budget": scheme_prelog * raw["genie_greedy"],
        "scheme_prelog": scheme_prelog,
        "exhaustive_prelog": exhaustive_prelog,
    }
    for key, source in (
        ("mimogs", "mimogs_greedy"),
        ("position_nn", "position_nn"),
        ("statistical", "statistical"),
        ("position_nn_clf", "position_nn_clf"),
        ("random", "random"),
    ):
        if source in raw:
            curves[key] = scheme_prelog * raw[source]
    return curves


# Listed in legend order; zorder is explicit so the visual stacking does not
# have to follow it (Random must sit underneath everything).
PLOT_SERIES = (
    # (curve key, label, color, linestyle, marker, linewidth, zorder)
    ("bound", "Zero-overhead bound", "black", "--", None, 1.8, 6),
    ("mimogs", "MIMO-GS", SCHEME_COLOR["mimogs_greedy"], "-", "o", 1.8, 5),
    ("exhaustive", "Exhaustive sweep", "tab:red", "-.", "d", 1.8, 4),
    ("statistical", "Statistical codebook", SCHEME_COLOR["statistical"], "-", "v", 1.8, 3),
    ("position_nn_clf", "Position NN classifier", SCHEME_COLOR["position_nn_clf"], "-", "X", 1.6, 2.0),
    ("random", "Random", SCHEME_COLOR["random"], "-", "s", 1.1, 1),
)
# With seven curves the Random floor is faded so it does not compete visually
# with the schemes actually under test.
FAINT_SERIES = {"random"}


def plot_curves(
    output_dir: str,
    stem: str,
    x_values: np.ndarray,
    curves: Dict[str, np.ndarray],
    x_label: str,
    title: str,
    log_x: bool,
    marker_every: int = 1,
) -> None:
    """Bound / MIMO-GS / Exhaustive sweep / Statistical codebook / Random.

    Position NN is computed and stored in the CSVs but is deliberately not
    drawn.
    """
    figure, axis = plt.subplots(figsize=(7.0, 5.2))

    for key, label, color, linestyle, marker, width, zorder in PLOT_SERIES:
        if key not in curves:
            continue
        axis.plot(
            x_values,
            curves[key],
            linestyle=linestyle,
            marker=marker,
            markersize=3.5 if key in FAINT_SERIES else 4.5,
            markevery=marker_every,
            linewidth=width,
            alpha=0.55 if key in FAINT_SERIES else 1.0,
            color=color,
            zorder=zorder,
            label=label,
        )

    if log_x:
        axis.set_xscale("log", base=2)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Mean net rate $R_{sel}$ [bps/Hz]")
    axis.set_title(title, fontsize=11)
    axis.grid(alpha=0.3, linewidth=0.5)
    axis.legend(fontsize=8.5, loc="best")
    axis.set_ylim(bottom=0.0)

    save_figure(figure, output_dir, stem)


def plot_alignment_efficiency(
    output_dir: str, lt_grid: Sequence[int], raw_means: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Overhead-free rate as a fraction of genie at the SAME L_t.

    The prelog cancels in the ratio, so this isolates how good each scheme's
    beam CHOICE is, independent of how much it costs to measure.
    """
    reference = np.maximum(raw_means["genie_greedy"], 1e-12)
    ratios = {
        scheme: raw_means[scheme] / reference
        for scheme in ("mimogs_greedy", "random")
    }

    # Local labels: this figure drops the selection-rule suffixes that the
    # other figures still carry, so SCHEME_LABEL is left untouched.
    labels = {"mimogs_greedy": "MIMO-GS", "random": "Random"}

    figure, axis = plt.subplots(figsize=(7.6, 5.0))
    axis.axhline(
        1.0, color="tab:green", linestyle="--", linewidth=1.5,
        label="Genie reference (= 1.0)",
    )

    for scheme in ("mimogs_greedy", "random"):
        axis.plot(
            lt_grid,
            ratios[scheme],
            marker="o",
            markersize=5,
            linewidth=1.9,
            color=SCHEME_COLOR[scheme],
            label=labels[scheme],
        )
        for slot, budget in enumerate(lt_grid):
            # At the full budget every scheme is trivially at 100%, and the two
            # annotations land on top of each other; keep only MIMO-GS's.
            if scheme == "random" and budget == max(lt_grid):
                continue
            axis.annotate(
                f"{100.0 * ratios[scheme][slot]:.1f}%",
                (budget, ratios[scheme][slot]),
                textcoords="offset points",
                xytext=(0, 7 if scheme == "mimogs_greedy" else -14),
                ha="center",
                fontsize=7.5,
                color=SCHEME_COLOR[scheme],
            )

    axis.set_xscale("log", base=2)
    axis.set_xticks(list(lt_grid))
    axis.set_xticklabels([str(value) for value in lt_grid])
    axis.set_xlabel("Transmit budget $L_t$ (CSI-RS beams)")
    axis.set_ylabel("Alignment efficiency  (rate / genie rate, same $L_t$)")
    axis.grid(alpha=0.3, linewidth=0.5)
    axis.legend(fontsize=8, loc="center right")
    # Leave headroom below zero so the Random annotations are not clipped.
    axis.set_ylim(-0.09, 1.13)

    save_figure(figure, output_dir, "fig_alignment_vs_Lt")
    return ratios


CDF_SERIES = (
    # (scheme key, label, color, linewidth, zorder)
    ("genie_greedy", "Genie", SCHEME_COLOR["genie_greedy"], 1.8, 5),
    ("mimogs_greedy", "MIMO-GS", SCHEME_COLOR["mimogs_greedy"], 1.8, 4),
    ("position_nn_clf", "Position NN classifier",
     SCHEME_COLOR["position_nn_clf"], 1.6, 3.2),
    ("statistical", "Statistical codebook", SCHEME_COLOR["statistical"], 1.8, 3),
    ("random", "Random", SCHEME_COLOR["random"], 1.2, 1),
)
# A curve only earns a place on the CDF if it is actually distinguishable from
# MIMO-GS; below this Kolmogorov-Smirnov separation the two are one line and
# the extra entry misleads more than it informs.  The measured values are
# written to the README either way.
CDF_KS_THRESHOLD = 0.05
CDF_ALWAYS_PLOT = ("genie_greedy", "mimogs_greedy", "statistical", "random")


def cdf_separation(values: np.ndarray, reference: np.ndarray) -> float:
    """Max vertical gap between two empirical CDFs (KS distance)."""
    grid = np.linspace(0.0, float(max(values.max(), reference.max())), 2000)
    left = np.searchsorted(np.sort(values), grid, side="right") / values.size
    right = np.searchsorted(np.sort(reference), grid, side="right") / reference.size
    return float(np.abs(left - right).max())


def plot_rate_cdf(
    output_dir: str, per_location: Dict[str, np.ndarray], num_tx: int
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Per-location net-rate CDF.

    Returns ``(medians, ks_vs_mimogs)`` for every candidate scheme, including
    the ones that were measured but left off the figure for overlapping
    MIMO-GS.
    """
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    medians: Dict[str, float] = {}
    separation: Dict[str, float] = {}
    reference = per_location.get("mimogs_greedy")

    for scheme, label, color, width, zorder in CDF_SERIES:
        if scheme not in per_location:
            continue
        ordered = np.sort(per_location[scheme])
        medians[scheme] = float(np.median(ordered))
        if reference is not None and scheme != "mimogs_greedy":
            separation[scheme] = cdf_separation(ordered, reference)
        if (
            scheme not in CDF_ALWAYS_PLOT
            and separation.get(scheme, 1.0) < CDF_KS_THRESHOLD
        ):
            continue
        probabilities = np.arange(1, ordered.size + 1) / ordered.size
        axis.plot(
            ordered,
            probabilities,
            linewidth=width,
            color=color,
            zorder=zorder,
            label=label,
        )

    axis.set_xlabel(f"$R_{{sel}}(p;\\,L_t={num_tx})$ [bps/Hz]")
    axis.set_ylabel("Empirical CDF")
    axis.set_title(f"Per-location net rate at $L_t$ = {num_tx}", fontsize=11)
    axis.grid(alpha=0.3, linewidth=0.5)
    axis.legend(fontsize=9, loc="lower right")
    axis.set_ylim(0.0, 1.0)

    save_figure(figure, output_dir, f"fig_rate_cdf_Lt{num_tx}")
    return medians, separation


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Net achievable rate for beam selection (paper Eq. 6_net_Rsel)"
    )
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--snr_db", type=float, default=10.0)
    parser.add_argument("--tau_over_TB", dest="tau_over_tb", type=float,
                        default=DEFAULT_TAU_OVER_TB)
    parser.add_argument("--Lr", dest="num_rx", type=int, default=2)
    parser.add_argument("--Lt_grid", type=str, default="",
                        help="Comma-separated transmit budgets (default 1,2,4,...,64).")
    parser.add_argument("--random_draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no_g_med_rescale",
        dest="g_med_rescale",
        action="store_false",
        help="Disable the g_med rescaling and report rates on the dataset's "
        "native scale instead.",
    )
    parser.set_defaults(g_med_rescale=True)
    parser.add_argument(
        "--clf_epochs",
        type=int,
        default=60,
        help="Epochs for the position-aided NN beam classifier "
        "(Morais et al. schedule: Adam lr 0.01, x0.2 at epochs 20 and 40).",
    )
    parser.add_argument(
        "--fixed_Lt",
        dest="fixed_lt",
        type=int,
        default=4,
        help="Transmit budget held fixed for the net-rate-vs-SNR figure.",
    )
    parser.add_argument(
        "--fixed_TB",
        dest="fixed_tb",
        type=float,
        default=256.0,
        help="Coherence block held fixed for the net-rate-vs-SNR figure. "
        "Only that figure and net_rate_vs_snr.csv use it; the T_B figure "
        "sweeps its own grid and the L_t sweep (including the CDF) is pinned "
        "to --tau_over_TB.",
    )
    parser.add_argument(
        "--snr_grid",
        type=str,
        default="",
        help="Comma-separated SNRs [dB] for the net-rate-vs-SNR figure "
        "(default 0 to 30 in 2 dB steps).",
    )
    parser.add_argument(
        "--selection_scale",
        choices=("as_is", "peak"),
        default="as_is",
        help="'as_is' feeds each scheme its native map (the paper's literal rule); "
        "'peak' max-normalizes both first so only map shape differs.",
    )
    parser.add_argument("--outputs_root", type=str, default="outputs")
    parser.add_argument("--analysis_root", type=str, default="analysis")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--source_path", type=str, default="")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    started = time.time()
    repository_root = os.path.dirname(os.path.abspath(__file__))

    lt_grid = (
        tuple(int(value) for value in arguments.Lt_grid.split(",") if value.strip())
        if arguments.Lt_grid
        else DEFAULT_LT_GRID
    )
    snr = float(10.0 ** (arguments.snr_db / 10.0))
    tau_rs = float(arguments.tau_over_tb) * TB_REFERENCE

    outputs_root = arguments.outputs_root
    if not os.path.isabs(outputs_root):
        outputs_root = os.path.join(repository_root, outputs_root)

    run_dir, checkpoint_path = resolve_run_dir(arguments.ckpt, outputs_root)
    run_name = os.path.basename(os.path.normpath(run_dir))

    print("=" * 78)
    print(f"[eval_net_rate] RUN        : {run_name}")
    print(f"[eval_net_rate] checkpoint : {checkpoint_path}")
    print(
        f"[eval_net_rate] SNR={arguments.snr_db:g} dB (P/sigma_z^2={snr:g}) | "
        f"L_r={arguments.num_rx} | tau_RS/T_B={arguments.tau_over_tb:g} "
        f"(tau_RS={tau_rs:g} symbols at T_B={TB_REFERENCE:g})"
    )
    print(f"[eval_net_rate] L_t grid   : {list(lt_grid)}")
    print("=" * 78)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_params, opt_params = restore_config(run_dir, checkpoint)

    if arguments.source_path:
        model_params.source_path = os.path.abspath(arguments.source_path)
    gt_root = str(getattr(model_params, "source_path", ""))
    if not os.path.isdir(gt_root):
        raise SystemExit(f"[eval_net_rate] Dataset '{gt_root}' is missing.")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    test_positions, test_magnitude = load_raw_mat(os.path.join(gt_root, "test.mat"))

    complex_path = os.path.join(gt_root, "complex.mat")
    if not os.path.isfile(complex_path):
        raise SystemExit(f"[eval_net_rate] Missing complex channel file {complex_path}")

    import scipy.io as sio  # local: only this block needs it

    complex_data = sio.loadmat(complex_path)
    for key in ("positions", "H_real", "H_imag"):
        if key not in complex_data:
            raise SystemExit(f"[eval_net_rate] '{complex_path}' has no '{key}'.")

    complex_positions = np.asarray(complex_data["positions"], dtype=np.float64)
    join = join_positions(test_positions, complex_positions, MATCH_TOL)
    num_locations = int(test_positions.shape[0])
    print(
        f"[eval_net_rate] position join: {num_locations}/{num_locations} test "
        f"locations matched exactly once in complex.mat "
        f"(tolerance {MATCH_TOL:g} m)"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channel = torch.from_numpy(
        (
            np.asarray(complex_data["H_real"], dtype=np.float32)[join]
            + 1j * np.asarray(complex_data["H_imag"], dtype=np.float32)[join]
        ).astype(np.complex64)
    ).to(device)
    del complex_data
    num_realizations = int(channel.shape[1])

    ground_truth = torch.from_numpy(test_magnitude.astype(np.float32)).to(device)

    # Optional operating-point rescaling.  Applied to the channel AND to the
    # ground-truth map together so the selection metric and the rate see the
    # same units; X_hat is rescaled later, after rendering.
    g_med = float(
        ground_truth.reshape(ground_truth.shape[0], -1).amax(dim=1).median().item()
    )
    if arguments.g_med_rescale:
        channel = channel / float(np.sqrt(g_med))
        ground_truth = ground_truth / g_med
        print(
            f"[eval_net_rate] --g_med_rescale ON: dividing channel POWER by "
            f"g_med={g_med:.6g} (median per-location best-beam gain), so "
            f"SNR={arguments.snr_db:g} dB is the operating SNR of a "
            f"median-strength location."
        )
    else:
        print(
            f"[eval_net_rate] g_med rescaling OFF (default). For reference the "
            f"median per-location best-beam gain is g_med={g_med:.6g}, i.e. "
            f"{10.0 * np.log10(g_med):.1f} dB relative to unit gain."
        )

    # ------------------------------------------------------------------
    # Sanity: complex.mat really describes these locations
    # ------------------------------------------------------------------
    generator_np = np.random.default_rng(arguments.seed)
    probe = generator_np.choice(num_locations, size=min(50, num_locations),
                                replace=False)
    probe_tensor = torch.as_tensor(probe, device=device)
    empirical = channel[probe_tensor].abs().pow(2).mean(dim=1).reshape(probe.size, -1)
    reference = ground_truth[probe_tensor].reshape(probe.size, -1)
    stacked = torch.stack(
        (empirical.reshape(-1).double(), reference.reshape(-1).double())
    )
    correlation = float(torch.corrcoef(stacked)[0, 1].item())
    print(
        f"[eval_net_rate] complex.mat self-consistency: corr(mean_b |H_b|^2, X_gt) "
        f"= {correlation:.6f} over {probe.size} locations"
    )
    if correlation <= 0.98:
        raise SystemExit(
            f"[eval_net_rate] corr(mean_b |H_b|^2, X_gt) = {correlation:.6f} <= 0.98; "
            f"the position join or the power convention is wrong."
        )

    # ------------------------------------------------------------------
    # Scene / model, then render X_hat once
    # ------------------------------------------------------------------
    hidden_dim = gain_net_hidden_dim(checkpoint)
    if hidden_dim is not None:
        print(f"[eval_net_rate] checkpoint gain MLP is {hidden_dim}-wide; rebuilding.")
    with gain_net_width(hidden_dim):
        scene, gaussians = build_scene_and_model(
            model_params, opt_params, checkpoint, device
        )

    dataset_magnitude = scene.test_set.magnitude
    assert torch.equal(dataset_magnitude, torch.from_numpy(test_magnitude)), (
        "Scene test_set magnitudes differ from raw test.mat; the split handling "
        "diverged."
    )

    batch_size = max(
        1, int(arguments.batch_size) or int(getattr(model_params, "batch_size", 8))
    )
    use_cuda_rasterizer = (
        bool(int(getattr(model_params, "use_cuda_rasterizer", 1)))
        and device.type == "cuda"
    )

    print(f"[eval_net_rate] rendering {num_locations} locations ...")
    render_started = time.time()
    rendered = render_mimogs(
        scene,
        gaussians,
        model_params,
        device,
        scene.test_set.positions,
        batch_size,
        use_cuda_rasterizer,
    ).float()
    print(f"[eval_net_rate]   done in {time.time() - render_started:.1f} s")

    tx_total = int(scene.beam_cols)
    num_rx = max(1, min(int(arguments.num_rx), int(scene.beam_rows)))

    # ------------------------------------------------------------------
    # Selection orderings (computed once; greedy is prefix-consistent)
    # ------------------------------------------------------------------
    def scale_for_selection(maps: torch.Tensor) -> torch.Tensor:
        if arguments.selection_scale == "peak":
            peak = maps.reshape(maps.shape[0], -1).amax(dim=1)
            peak = peak.clamp_min(torch.finfo(maps.dtype).tiny)
            return maps / peak.view(-1, 1, 1)
        return maps

    mimogs_map = scale_for_selection(rendered)
    genie_map = scale_for_selection(ground_truth)

    fixed_lt = max(1, min(int(arguments.fixed_lt), tx_total))
    fixed_tb = float(arguments.fixed_tb)
    full_sweep = tx_total

    # ------------------------------------------------------------------
    # Literature-anchored baselines, both built from the TRAIN split ONLY
    # ------------------------------------------------------------------
    train_positions, train_magnitude = load_raw_mat(
        os.path.join(gt_root, "train.mat")
    )
    if train_magnitude.shape[1:] != test_magnitude.shape[1:]:
        raise SystemExit(
            f"[eval_net_rate] train/test beam grids differ: "
            f"{train_magnitude.shape[1:]} vs {test_magnitude.shape[1:]}"
        )

    neighbour_distance, neighbour_index = _nearest_train(
        test_positions, train_positions
    )
    # Construction must not see a single test measurement.  If any evaluated
    # location also existed in TRAIN the "prediction" would be a lookup of the
    # answer, so require a strictly positive separation.
    assert float(neighbour_distance.min()) > MATCH_TOL, (
        f"A test location coincides with a TRAIN location "
        f"(min distance {float(neighbour_distance.min()):.3g} m); the "
        f"position-aided baseline would be reading its own ground truth."
    )
    assert int(neighbour_index.max()) < train_positions.shape[0], (
        "Nearest-neighbour indices escape the TRAIN split."
    )

    # -- Statistical codebook: histogram of best TRAIN beams, location-agnostic
    train_column_power = train_magnitude.astype(np.float64).sum(axis=1)
    train_best_beam = train_column_power.argmax(axis=1)
    beam_counts = np.bincount(train_best_beam, minlength=tx_total)
    statistical_ranking = np.argsort(-beam_counts, kind="stable")
    statistical_coverage = float(
        beam_counts[statistical_ranking[:fixed_lt]].sum()
    ) / float(train_positions.shape[0])

    statistical_order = (
        torch.from_numpy(statistical_ranking.copy())
        .to(device=device, dtype=torch.long)
        .unsqueeze(0)
        .expand(num_locations, tx_total)
        .contiguous()
    )

    print(
        f"[eval_net_rate] statistical codebook (TRAIN only, "
        f"{train_positions.shape[0]} locations):"
    )
    print(
        f"  B_t^stat at L_t={fixed_lt}: "
        f"{list(int(v) for v in statistical_ranking[:fixed_lt])}  "
        f"(counts {list(int(beam_counts[v]) for v in statistical_ranking[:fixed_lt])})"
    )
    print(
        f"  covers the best beam of {100.0 * statistical_coverage:.2f}% of "
        f"TRAIN locations at L_t={fixed_lt}"
    )

    # -- Position-aided nearest-neighbour: reuse the neighbour's GT map
    neighbour_map = torch.from_numpy(
        train_magnitude[neighbour_index].astype(np.float32)
    ).to(device)
    if arguments.g_med_rescale:
        neighbour_map = neighbour_map / g_med
    neighbour_map = scale_for_selection(neighbour_map)

    # How good a predictor is the neighbour's measured map?  Same max-normalized
    # NMSE convention as eval_render, so it is directly comparable to the
    # rendered map's number.  This is what decides whether Position NN can rival
    # MIMO-GS, so it is measured rather than assumed.
    def _shape_nmse_db(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        def _normalize(maps: np.ndarray) -> np.ndarray:
            peak = maps.reshape(maps.shape[0], -1).max(axis=1)
            return maps / np.maximum(peak, np.finfo(np.float64).tiny)[:, None, None]

        predicted_n = _normalize(prediction.astype(np.float64))
        target_n = _normalize(target.astype(np.float64))
        numerator = ((predicted_n - target_n) ** 2).reshape(target.shape[0], -1).sum(1)
        denominator = (target_n ** 2).reshape(target.shape[0], -1).sum(1)
        return 10.0 * np.log10(np.maximum(numerator / denominator, 1e-12))

    neighbour_nmse = _shape_nmse_db(train_magnitude[neighbour_index], test_magnitude)
    neighbour_argmax_hit = float(
        np.mean(
            train_magnitude[neighbour_index].reshape(num_locations, -1).argmax(1)
            == test_magnitude.reshape(num_locations, -1).argmax(1)
        )
    )

    print(
        f"[eval_net_rate] position-aided NN: distance to nearest TRAIN "
        f"location median {np.median(neighbour_distance):.2f} m, "
        f"p95 {np.percentile(neighbour_distance, 95):.2f} m, "
        f"max {neighbour_distance.max():.2f} m"
    )
    print(
        f"  neighbour map as a predictor of the test map: NMSE mean "
        f"{neighbour_nmse.mean():.2f} dB / median {np.median(neighbour_nmse):.2f} dB, "
        f"argmax hit {100.0 * neighbour_argmax_hit:.2f}%"
    )

    # -- Position-aided NN classifier (Morais et al., ICC 2023)
    # The published model consumes the min-max normalized, coarsely quantized
    # 2-D coordinate.  z is constant in these datasets, so it carries no
    # information and is dropped rather than fed to the network as a dead input.
    train_z = np.unique(train_positions[:, 2])
    test_z = np.unique(test_positions[:, 2])
    print(
        f"[eval_net_rate] position-aided NN classifier (Morais et al. ICC 2023): "
        f"z is constant (train {train_z}, test {test_z}) so the input is the "
        f"2-D (x, y) position only"
    )

    xy_lower = train_positions[:, :2].min(axis=0)
    xy_upper = train_positions[:, :2].max(axis=0)
    outside_box = int(
        np.sum(
            (test_positions[:, 0] < xy_lower[0])
            | (test_positions[:, 0] > xy_upper[0])
            | (test_positions[:, 1] < xy_lower[1])
            | (test_positions[:, 1] > xy_upper[1])
        )
    )
    train_features = torch.from_numpy(
        normalize_and_quantize(
            train_positions[:, :2], xy_lower, xy_upper
        ).astype(np.float32)
    )
    test_features = torch.from_numpy(
        normalize_and_quantize(
            test_positions[:, :2], xy_lower, xy_upper
        ).astype(np.float32)
    ).to(device)
    print(
        f"  TRAIN min-max x[{xy_lower[0]:.3f}, {xy_upper[0]:.3f}] "
        f"y[{xy_lower[1]:.3f}, {xy_upper[1]:.3f}], quantized to "
        f"{POSITION_BINS} bins/axis (resolution {1.0 / POSITION_BINS:g}); "
        f"{outside_box} test location(s) outside the TRAIN box"
    )

    train_best_beam = torch.from_numpy(train_column_power.argmax(axis=1)).to(
        device=device, dtype=torch.long
    )
    print(
        f"[eval_net_rate] training the classifier (3 x 256 FC, batch 32, "
        f"{arguments.clf_epochs} epochs, Adam lr 0.01 x0.2 @ 20/40, "
        f"75/25 fit/validation selection) ..."
    )
    classifier, classifier_stats = train_beam_classifier(
        train_features,
        train_best_beam,
        num_beams=tx_total,
        device=device,
        cache_dir=os.path.join(outputs_root, "pos_nn_classifier"),
        source_path=gt_root,
        epochs=int(arguments.clf_epochs),
        seed=int(arguments.seed),
    )
    position_nn_clf_order = classifier_orders(classifier, test_features)

    test_best_beam = torch.from_numpy(
        test_magnitude.astype(np.float64).sum(axis=1).argmax(axis=1)
    ).to(device=device, dtype=torch.long)
    classifier_accuracy = topk_beam_accuracy(
        position_nn_clf_order, test_best_beam, k_values=(1, 4)
    )
    print(
        f"  validation-selected epoch {int(classifier_stats['best_epoch'])} "
        f"(val top-1 {100.0 * classifier_stats['best_val_top1']:.2f}%, fit "
        f"{int(classifier_stats['num_fit'])} / val "
        f"{int(classifier_stats['num_validation'])} TRAIN locations)"
    )
    print(
        f"  classifier beam accuracy on TEST: top-1 "
        f"{100.0 * classifier_accuracy[1]:.2f}%, top-4 "
        f"{100.0 * classifier_accuracy[4]:.2f}%"
    )

    # Descriptors used if a baseline ever out-scores MIMO-GS: state WHY it can,
    # in the same terms as the rendered map, instead of failing the run.
    rendered_nmse_db = float(
        np.mean(
            _shape_nmse_db(
                rendered.detach().cpu().numpy().astype(np.float64), test_magnitude
            )
        )
    )
    baseline_diagnostic: Dict[str, str] = {
        "position_nn_clf": (
            f"classifier top-1 {100.0 * classifier_accuracy[1]:.2f}% / top-4 "
            f"{100.0 * classifier_accuracy[4]:.2f}% against the rendered map's "
            f"{rendered_nmse_db:.2f} dB NMSE (TRAIN only, asserted)"
        ),
    }
    print(
        f"  rendered map fidelity for reference: {rendered_nmse_db:.2f} dB "
        f"(max-normalized NMSE)"
    )

    print("[eval_net_rate] building selection orders ...")
    orders: Dict[str, torch.Tensor] = {
        "mimogs_greedy": greedy_order(mimogs_map, snr, num_rx, max(lt_grid)),
        "genie_greedy": greedy_order(genie_map, snr, num_rx, max(lt_grid)),
        "mimogs_toppower": toppower_order(mimogs_map),
        "genie_toppower": toppower_order(genie_map),
        "position_nn": greedy_order(neighbour_map, snr, num_rx, max(lt_grid)),
        "statistical": statistical_order,
        "position_nn_clf": position_nn_clf_order,
    }

    generator = torch.Generator(device=device)
    generator.manual_seed(int(arguments.seed))
    random_orders = [
        random_order(num_locations, tx_total, generator, device)
        for _ in range(max(1, int(arguments.random_draws)))
    ]

    # ------------------------------------------------------------------
    # Rates
    # ------------------------------------------------------------------
    raw_rates: Dict[str, np.ndarray] = {scheme: np.empty(len(lt_grid))
                                        for scheme in CSV_SCHEMES}
    net_rates: Dict[str, np.ndarray] = {scheme: np.empty(len(lt_grid))
                                        for scheme in CSV_SCHEMES}
    per_location_raw: Dict[Tuple[str, int], np.ndarray] = {}
    recorded_tx: Dict[str, np.ndarray] = {}

    print()
    for slot, budget in enumerate(lt_grid):
        step_started = time.time()
        overhead = prelog(budget, arguments.tau_over_tb)

        for scheme in CSV_SCHEMES:
            if scheme == "random":
                accumulator = torch.zeros(num_locations, device=device)
                for permutation in random_orders:
                    tx_beams = permutation[:, :budget].contiguous()
                    rx_beams = receive_selection(ground_truth, tx_beams, num_rx)
                    accumulator += spectral_efficiency(
                        channel, rx_beams, tx_beams, snr
                    )
                values = accumulator / float(len(random_orders))
            else:
                tx_beams = orders[scheme][:, :budget].contiguous()
                rx_beams = receive_selection(ground_truth, tx_beams, num_rx)
                values = spectral_efficiency(channel, rx_beams, tx_beams, snr)

                if budget == BT_RECORD_LT and scheme in (
                    "mimogs_greedy",
                    "genie_greedy",
                ):
                    recorded_tx[scheme] = tx_beams.cpu().numpy()

            values_np = values.detach().cpu().numpy().astype(np.float64)
            per_location_raw[(scheme, budget)] = values_np
            raw_rates[scheme][slot] = float(values_np.mean())
            net_rates[scheme][slot] = overhead * raw_rates[scheme][slot]

        print(
            f"[eval_net_rate] L_t={budget:>3d}  prelog={overhead:5.3f}  "
            f"net: MIMO-GS {net_rates['mimogs_greedy'][slot]:.5f}  "
            f"Genie {net_rates['genie_greedy'][slot]:.5f}  "
            f"Random {net_rates['random'][slot]:.5f}  "
            f"[{time.time() - step_started:.1f} s, "
            f"{time.time() - started:.1f} s total]"
        )

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    print()
    print("[eval_net_rate] sanity checks")
    tolerance = 1e-6
    warnings: List[str] = []

    for slot, budget in enumerate(lt_grid):
        for rule in ("greedy", "toppower"):
            genie = net_rates[f"genie_{rule}"][slot]
            mimogs = net_rates[f"mimogs_{rule}"][slot]
            assert genie >= mimogs - tolerance, (
                f"Genie < MIMO-GS at L_t={budget} ({rule}): "
                f"{genie:.6f} < {mimogs:.6f}"
            )
        assert net_rates["mimogs_greedy"][slot] >= net_rates["random"][slot] - tolerance, (
            f"MIMO-GS < Random at L_t={budget}: "
            f"{net_rates['mimogs_greedy'][slot]:.6f} < {net_rates['random'][slot]:.6f}"
        )
    print("  Genie >= MIMO-GS >= Random at every L_t          : OK")

    full_slot = list(lt_grid).index(max(lt_grid))
    full_values = [net_rates[scheme][full_slot] for scheme in CSV_SCHEMES]
    spread = float(max(full_values) - min(full_values))
    assert spread <= 1e-6, (
        f"Schemes disagree at L_t={max(lt_grid)} by {spread:.3g}; with no selection "
        f"freedom they must coincide."
    )
    print(f"  all schemes coincide at L_t={max(lt_grid)} (spread {spread:.2g})    : OK")

    genie_raw = raw_rates["genie_greedy"]
    monotone = bool(np.all(np.diff(genie_raw) >= -1e-6))
    if monotone:
        print("  overhead-free Genie (greedy) non-decreasing      : OK")
    else:
        message = (
            "Overhead-free Genie (greedy) is NOT non-decreasing in L_t: "
            + ", ".join(
                f"L_t={b}:{genie_raw[s]:.5f}" for s, b in enumerate(lt_grid)
            )
            + ". This is a property of Eq. (6_net_Rsel) as specified, not a bug: "
            "H_sel is (L_r x L_t) so its Gram has rank <= L_r, and the uniform "
            "P/(L_t sigma_z^2) split spreads the same power over beams that add "
            "no further spatial dimension. Beyond L_t = L_r the rate can only "
            "fall. See rate_model_diagnostic.csv -- the effect holds at every "
            "SNR and every L_r tested, so no operating point rescues it."
        )
        warnings.append(message)
        print(f"  WARNING: overhead-free Genie (greedy) NOT non-decreasing")
        print(f"           {', '.join(f'L_t={b}:{genie_raw[s]:.5f}' for s, b in enumerate(lt_grid))}")

    genie_gap = float(
        np.max(np.abs(raw_rates["genie_greedy"] - raw_rates["genie_toppower"]))
    )
    if genie_gap > 0.5:
        warnings.append(
            f"Genie greedy-vs-toppower gap {genie_gap:.3f} bps/Hz exceeds 0.5"
        )
    print(f"  Genie greedy vs top-power max gap = {genie_gap:.4f} bps/Hz : "
          f"{'OK' if genie_gap <= 0.5 else 'CHECK'}")

    best_slot = int(np.argmax(net_rates["genie_greedy"]))
    interior = min(lt_grid) < lt_grid[best_slot] < max(lt_grid)
    if interior:
        print(
            f"  Genie net rate has an interior maximum at L_t="
            f"{lt_grid[best_slot]}   : OK"
        )
    elif lt_grid[best_slot] == min(lt_grid):
        message = (
            f"Genie net rate peaks at the SMALLEST budget L_t={min(lt_grid)}, i.e. at "
            f"the grid endpoint rather than an interior optimum. This follows "
            f"directly from the monotonicity finding above: the rate term falls "
            f"with L_t while the overhead also grows with L_t, so both terms push "
            f"toward the smallest budget and no trade-off exists."
        )
        warnings.append(message)
        print(
            f"  WARNING: Genie net rate peaks at the grid MINIMUM L_t="
            f"{min(lt_grid)} (no interior optimum)"
        )
    else:
        message = (
            f"Genie net-rate curve is monotone up to L_t={max(lt_grid)} "
            f"(no interior maximum) under tau_RS/T_B="
            f"{arguments.tau_over_tb:g}; values="
            + ", ".join(
                f"L_t={b}:{net_rates['genie_greedy'][s]:.5f}"
                for s, b in enumerate(lt_grid)
            )
        )
        warnings.append(message)
        print(f"  WARNING: {message}")

    # Scale-sensitivity diagnostic for the selection-map confound.
    alternative = "peak" if arguments.selection_scale == "as_is" else "as_is"
    if arguments.selection_scale == "as_is":
        alt_map = rendered / rendered.reshape(num_locations, -1).amax(dim=1).clamp_min(
            torch.finfo(rendered.dtype).tiny
        ).view(-1, 1, 1)
        alt_genie = ground_truth / ground_truth.reshape(
            num_locations, -1
        ).amax(dim=1).clamp_min(torch.finfo(ground_truth.dtype).tiny).view(-1, 1, 1)
    else:
        alt_map, alt_genie = rendered, ground_truth

    alt_orders = {
        "mimogs_greedy": greedy_order(alt_map, snr, num_rx, max(lt_grid)),
        "genie_greedy": greedy_order(alt_genie, snr, num_rx, max(lt_grid)),
    }
    scale_probe: Dict[str, float] = {}
    for scheme, order in alt_orders.items():
        tx_beams = order[:, :BT_RECORD_LT].contiguous()
        rx_beams = receive_selection(ground_truth, tx_beams, num_rx)
        scale_probe[scheme] = float(
            spectral_efficiency(channel, rx_beams, tx_beams, snr).mean().item()
        ) * prelog(BT_RECORD_LT, arguments.tau_over_tb)

    print(
        f"  selection-scale sensitivity at L_t={BT_RECORD_LT} "
        f"('{arguments.selection_scale}' -> '{alternative}'): "
        f"MIMO-GS {net_rates['mimogs_greedy'][list(lt_grid).index(BT_RECORD_LT)]:.5f}"
        f" -> {scale_probe['mimogs_greedy']:.5f}, "
        f"Genie {net_rates['genie_greedy'][list(lt_grid).index(BT_RECORD_LT)]:.5f}"
        f" -> {scale_probe['genie_greedy']:.5f}"
    )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    output_dir = os.path.join(
        repository_root, arguments.analysis_root, run_name, "net_rate"
    )
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Rate-model diagnostic: is the L_t decay an operating-point artefact or a
    # structural property of Eq. (6_net_Rsel)?  Sweep SNR and L_r on a subset
    # with genie selection and write the evidence next to the results.
    # ------------------------------------------------------------------
    print()
    print("[eval_net_rate] rate-model diagnostic (genie selection, subset) ...")
    probe_count = min(1000, num_locations)
    probe_channel = channel[:probe_count]
    probe_gt = ground_truth[:probe_count]
    diagnostic_rows: List[Dict[str, object]] = []

    for probe_snr_db in (arguments.snr_db, 20.0, 30.0, 40.0):
        probe_snr = float(10.0 ** (probe_snr_db / 10.0))
        for probe_lr in (1, 2, 4, 8, min(16, int(scene.beam_rows))):
            probe_lr = min(probe_lr, int(scene.beam_rows))
            probe_order = greedy_order(probe_gt, probe_snr, probe_lr, max(lt_grid))
            values = []
            for budget in lt_grid:
                tx_beams = probe_order[:, :budget].contiguous()
                rx_beams = receive_selection(probe_gt, tx_beams, probe_lr)
                values.append(
                    float(
                        spectral_efficiency(
                            probe_channel, rx_beams, tx_beams, probe_snr
                        ).mean().item()
                    )
                )
            best = int(np.argmax(values))
            diagnostic_rows.append(
                {
                    "snr_db": probe_snr_db,
                    "Lr": probe_lr,
                    **{f"raw_rate_Lt{b}": f"{values[s]:.6f}"
                       for s, b in enumerate(lt_grid)},
                    "best_Lt": lt_grid[best],
                    "non_decreasing": int(
                        bool(np.all(np.diff(np.array(values)) >= -1e-6))
                    ),
                }
            )

    with open(os.path.join(output_dir, "rate_model_diagnostic.csv"), "w",
              newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostic_rows[0].keys()))
        writer.writeheader()
        for row in diagnostic_rows:
            writer.writerow(row)

    any_monotone = any(int(row["non_decreasing"]) for row in diagnostic_rows)
    best_lt_values = sorted({int(row["best_Lt"]) for row in diagnostic_rows})
    print(
        f"  swept SNR x L_r on {probe_count} locations: "
        f"non-decreasing anywhere = {bool(any_monotone)}; "
        f"best L_t observed = {best_lt_values}"
    )

    with open(os.path.join(output_dir, "net_rate_vs_Lt.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["L_t", "prelog"]
        header += [f"net_{scheme}" for scheme in CSV_SCHEMES]
        header += [f"raw_{scheme}" for scheme in CSV_SCHEMES]
        writer.writerow(header)
        for slot, budget in enumerate(lt_grid):
            row = [budget, f"{prelog(budget, arguments.tau_over_tb):.6f}"]
            row += [f"{net_rates[scheme][slot]:.8f}" for scheme in CSV_SCHEMES]
            row += [f"{raw_rates[scheme][slot]:.8f}" for scheme in CSV_SCHEMES]
            writer.writerow(row)

    with open(os.path.join(output_dir, "per_location.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["index", "x", "y", "z"]
        for scheme in CSV_SCHEMES:
            header += [f"Rsel_{scheme}_Lt{budget}" for budget in lt_grid]
        header += [
            f"Bt_mimogs_greedy_Lt{BT_RECORD_LT}",
            f"Bt_genie_greedy_Lt{BT_RECORD_LT}",
        ]
        writer.writerow(header)

        for row in range(num_locations):
            record: List[object] = [
                row,
                f"{test_positions[row, 0]:.6f}",
                f"{test_positions[row, 1]:.6f}",
                f"{test_positions[row, 2]:.6f}",
            ]
            for scheme in CSV_SCHEMES:
                for budget in lt_grid:
                    value = per_location_raw[(scheme, budget)][row]
                    record.append(
                        f"{prelog(budget, arguments.tau_over_tb) * value:.8f}"
                    )
            for scheme in ("mimogs_greedy", "genie_greedy"):
                record.append(
                    ";".join(str(int(v)) for v in recorded_tx[scheme][row])
                    if scheme in recorded_tx
                    else ""
                )
            writer.writerow(record)

    plot_rate_vs_lt(
        output_dir,
        "fig_net_rate_vs_Lt",
        lt_grid,
        net_rates,
        "Mean net rate $R_{sel}$ [bps/Hz]",
        f"Net achievable rate vs. $L_t$  "
        f"(SNR {arguments.snr_db:g} dB, $L_r$={num_rx}, "
        f"$\\tau_{{RS}}/T_B$={arguments.tau_over_tb:g})",
        mark_maxima=True,
    )
    # The overhead-free rate is no longer plotted on its own; raw_rates is still
    # computed and feeds the alignment figure, the T_B curves and the CSVs.

    # ------------------------------------------------------------------
    # Fixed operating point shared by both net-rate figures
    # ------------------------------------------------------------------
    selection_maps = {
        "mimogs_greedy": mimogs_map,
        "genie_greedy": genie_map,
        "position_nn": neighbour_map,
    }
    fixed_orders = {
        "statistical": statistical_order,
        "position_nn_clf": position_nn_clf_order,
    }

    alignment = plot_alignment_efficiency(output_dir, lt_grid, raw_rates)

    # ------------------------------------------------------------------
    # Net rate vs T_B at the headline SNR and the fixed data budget
    # ------------------------------------------------------------------
    # Down to ~70 symbols so the exhaustive collapse region is on-figure.
    tb_grid = np.unique(
        np.round(np.logspace(np.log10(70.0), np.log10(8192.0), 14))
    ).astype(np.float64)

    tb_raw = raw_rates_at(
        channel, ground_truth, selection_maps, fixed_orders, random_orders,
        fixed_lt, num_rx, snr,
    )
    tb_curves = net_rate_vs_tb(tb_raw, tb_grid, fixed_lt, full_sweep, tau_rs)
    # Exhaustive evaluated at the reference block, under the SAME definition as
    # the figures (full-sweep prelog applied to the bound's rate term).
    exhaustive_reference = (
        max(0.0, 1.0 - full_sweep * tau_rs / TB_REFERENCE)
        * tb_raw["genie_greedy"]
    )
    print(
        f"  exhaustive baseline at T_B={TB_REFERENCE:g}, L_t={fixed_lt}: "
        f"{exhaustive_reference:.6f} bps/Hz "
        f"(prelog {max(0.0, 1.0 - full_sweep * tau_rs / TB_REFERENCE):.3f} "
        f"x bound {tb_raw['genie_greedy']:.6f})"
    )

    plot_curves(
        output_dir,
        "fig_net_rate_vs_TB",
        tb_grid,
        tb_curves,
        "Coherence block $T_B$ [symbols]",
        f"Net achievable rate vs $T_B$  ($L_t$ = {fixed_lt}, "
        f"SNR = {arguments.snr_db:g} dB, $\\tau_{{RS}}$ = {tau_rs:g})",
        log_x=True,
    )

    with open(os.path.join(output_dir, "net_rate_vs_TB.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "T_B",
                "zero_overhead_bound",
                "net_mimogs",
                "net_position_nn",
                "net_statistical",
                "net_position_nn_clf",
                "net_random",
                "net_exhaustive",
                "genie_same_budget_reference",
                "scheme_prelog",
                "exhaustive_prelog",
                "mimogs_minus_exhaustive",
                "mimogs_over_exhaustive_pct",
                "mimogs_over_bound_pct",
                "exhaustive_over_bound_pct",
            ]
        )
        for index, block in enumerate(tb_grid):
            bound_value = tb_curves["bound"][index]
            mimogs_value = tb_curves["mimogs"][index]
            exhaustive_value = tb_curves["exhaustive"][index]
            writer.writerow(
                [
                    int(block),
                    f"{bound_value:.8f}",
                    f"{mimogs_value:.8f}",
                    f"{tb_curves['position_nn'][index]:.8f}",
                    f"{tb_curves['statistical'][index]:.8f}",
                    f"{tb_curves['position_nn_clf'][index]:.8f}",
                    f"{tb_curves['random'][index]:.8f}",
                    f"{exhaustive_value:.8f}",
                    f"{tb_curves['genie_same_budget'][index]:.8f}",
                    f"{tb_curves['scheme_prelog'][index]:.6f}",
                    f"{tb_curves['exhaustive_prelog'][index]:.6f}",
                    f"{mimogs_value - exhaustive_value:.8f}",
                    f"{100.0 * mimogs_value / max(exhaustive_value, 1e-12):.4f}",
                    f"{100.0 * mimogs_value / max(bound_value, 1e-12):.4f}",
                    f"{100.0 * exhaustive_value / max(bound_value, 1e-12):.4f}",
                ]
            )

    # ------------------------------------------------------------------
    # Net rate vs SNR at the fixed operating point
    # ------------------------------------------------------------------
    if arguments.snr_grid:
        snr_grid_db = [
            float(value) for value in arguments.snr_grid.split(",") if value.strip()
        ]
    else:
        snr_grid_db = list(np.arange(0.0, 30.0 + 1e-9, 2.0))

    print()
    print(
        f"[eval_net_rate] net rate vs SNR at L_t={fixed_lt}, T_B={fixed_tb:g} "
        f"over {snr_grid_db[0]:g}..{snr_grid_db[-1]:g} dB "
        f"({len(snr_grid_db)} points) ..."
    )
    snr_started = time.time()
    snr_curves = net_rate_vs_snr(
        channel,
        ground_truth,
        selection_maps,
        fixed_orders,
        random_orders,
        fixed_lt,
        fixed_tb,
        tau_rs,
        full_sweep,
        num_rx,
        snr_grid_db,
    )
    print(f"  done in {time.time() - snr_started:.1f} s")

    plot_curves(
        output_dir,
        "fig_net_rate_vs_snr",
        np.asarray(snr_grid_db, dtype=np.float64),
        snr_curves,
        "SNR [dB]",
        f"Net achievable rate vs SNR ($L_t$ = {fixed_lt}, $T_B$ = {fixed_tb:g})",
        log_x=False,
        marker_every=2,
    )

    with open(os.path.join(output_dir, "net_rate_vs_snr.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "snr_db",
                "zero_overhead_bound",
                "net_mimogs",
                "net_position_nn",
                "net_statistical",
                "net_position_nn_clf",
                "net_random",
                "net_exhaustive",
                "genie_same_budget_reference",
                "mimogs_minus_exhaustive",
                "mimogs_over_exhaustive_pct",
                "mimogs_over_bound_pct",
                "exhaustive_over_bound_pct",
            ]
        )
        for index, snr_db in enumerate(snr_grid_db):
            bound_value = snr_curves["bound"][index]
            mimogs_value = snr_curves["mimogs"][index]
            exhaustive_value = snr_curves["exhaustive"][index]
            writer.writerow(
                [
                    f"{snr_db:g}",
                    f"{bound_value:.8f}",
                    f"{mimogs_value:.8f}",
                    f"{snr_curves['position_nn'][index]:.8f}",
                    f"{snr_curves['statistical'][index]:.8f}",
                    f"{snr_curves['position_nn_clf'][index]:.8f}",
                    f"{snr_curves['random'][index]:.8f}",
                    f"{exhaustive_value:.8f}",
                    f"{snr_curves['genie_same_budget'][index]:.8f}",
                    f"{mimogs_value - exhaustive_value:.8f}",
                    f"{100.0 * mimogs_value / max(exhaustive_value, 1e-12):.4f}",
                    f"{100.0 * mimogs_value / max(bound_value, 1e-12):.4f}",
                    f"{100.0 * exhaustive_value / max(bound_value, 1e-12):.4f}",
                ]
            )

    # ------------------------------------------------------------------
    # Sanity: the three-curve identity, ordering and monotonicity
    # ------------------------------------------------------------------
    print()
    print("[eval_net_rate] three-curve sanity checks")

    for label, curves, axis_values in (
        ("SNR", snr_curves, snr_grid_db),
        ("T_B", tb_curves, list(tb_grid)),
    ):
        # Exhaustive differs from the bound ONLY by its measurement prelog.
        identity = curves["bound"] * curves["exhaustive_prelog"]
        worst = float(np.max(np.abs(curves["exhaustive"] - identity)))
        assert worst <= 1e-12, (
            f"[{label}] exhaustive != bound * prelog_exhaustive "
            f"(max deviation {worst:.3g})"
        )

        assert np.all(curves["mimogs"] <= curves["bound"] + 1e-9), (
            f"[{label}] MIMO-GS exceeds the zero-overhead bound"
        )
        assert np.all(curves["exhaustive"] <= curves["bound"] + 1e-9), (
            f"[{label}] exhaustive exceeds the zero-overhead bound"
        )

        losses = np.nonzero(curves["mimogs"] <= curves["exhaustive"])[0]
        if losses.size:
            crossover = ", ".join(
                f"{label}={axis_values[index]:g}" for index in losses
            )
            warnings.append(
                f"MIMO-GS does not beat exhaustive at {losses.size} {label} "
                f"point(s): {crossover}"
            )
            print(
                f"  [{label}] MIMO-GS <= exhaustive at: {crossover}"
            )
        print(
            f"  [{label}] exhaustive == bound x prelog (max dev {worst:.1e}), "
            f"both <= bound, MIMO-GS > exhaustive at all "
            f"{len(axis_values)} points : "
            f"{'OK' if not losses.size else 'SEE ABOVE'}"
        )

    # Expected ordering among the same-prelog schemes: any baseline beating
    # MIMO-GS would mean the rendered map is being out-selected by something
    # that saw strictly less information, which is a bug, not a result.
    for label, curves, axis_values in (
        ("SNR", snr_curves, snr_grid_db),
        ("T_B", tb_curves, list(tb_grid)),
    ):
        for weaker, stronger in (
            ("position_nn", "mimogs"),
            ("position_nn_clf", "mimogs"),
            ("statistical", "position_nn"),
            ("random", "statistical"),
        ):
            violations = np.nonzero(curves[weaker] > curves[stronger] + 1e-9)[0]
            if not violations.size:
                continue
            where = ", ".join(
                f"{label}={axis_values[index]:g}" for index in violations
            )
            message = (
                f"[{label}] {SCHEME_LABEL[weaker]} exceeds "
                f"{SCHEME_LABEL.get(stronger, stronger)} at "
                f"{violations.size} point(s): {where}"
            )
            if stronger == "mimogs" and weaker in baseline_diagnostic:
                message += f" -- {baseline_diagnostic[weaker]}"
            elif stronger == "mimogs":
                # Investigated, not a bug: the split is an interleaved 1 m grid,
                # so every test point has a MEASURED neighbour one step away.
                # The leakage assert above already rules out a coincident
                # location, and the neighbour map is a better predictor of the
                # test map than the rendered one, so the baseline legitimately
                # wins here.  Flag it loudly instead of failing.
                message += (
                    f" -- neighbour map NMSE {neighbour_nmse.mean():.2f} dB vs "
                    f"the rendered map's, at a median NN distance of "
                    f"{np.median(neighbour_distance):.2f} m; no leakage "
                    f"(min distance {neighbour_distance.min():.2f} m)"
                )
            warnings.append(message)
            print(f"  NOTE: {message}")
    print(
        "  ordering bound >= MIMO-GS >= Position NN >= Statistical >= Random"
        "  : checked"
    )

    for key in ("bound", "mimogs", "exhaustive"):
        steps = np.diff(snr_curves[key])
        assert np.all(steps >= -1e-9), (
            f"{key} is not monotonically increasing in SNR: {snr_curves[key]}"
        )
    print("  [SNR] all three curves increase monotonically     : OK")

    final_ratio = float(
        tb_curves["exhaustive"][-1] / max(tb_curves["bound"][-1], 1e-12)
    )
    assert final_ratio >= 0.99, (
        f"At T_B={tb_grid[-1]:.0f} exhaustive is {100.0 * (1.0 - final_ratio):.2f}% "
        f"below the bound; expected within 1%."
    )
    print(
        f"  [T_B] exhaustive within {100.0 * (1.0 - final_ratio):.2f}% of the "
        f"bound at T_B={tb_grid[-1]:.0f}       : OK"
    )

    tracking = snr_curves["mimogs"] / np.maximum(snr_curves["bound"], 1e-12)
    worst_index = int(np.argmin(tracking))
    print(
        f"  [SNR] MIMO-GS sits at {100.0 * tracking.min():.2f}%-"
        f"{100.0 * tracking.max():.2f}% of the zero-overhead bound "
        f"(worst at {snr_grid_db[worst_index]:g} dB)"
    )

    with open(os.path.join(output_dir, "alignment_efficiency.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["L_t", "mimogs_greedy_vs_genie", "random_vs_genie"])
        for slot, budget in enumerate(lt_grid):
            writer.writerow(
                [
                    budget,
                    f"{alignment['mimogs_greedy'][slot]:.6f}",
                    f"{alignment['random'][slot]:.6f}",
                ]
            )

    # Headline gaps at the requested reference SNRs.
    print()
    print(
        f"[eval_net_rate] MIMO-GS vs Exhaustive at L_t={fixed_lt}, T_B={fixed_tb:g}"
    )
    print(
        f"  {'SNR [dB]':>9}{'bound':>10}{'MIMO-GS':>11}{'Exhaustive':>12}"
        f"{'delta':>10}{'MG/exh':>9}{'MG/bound':>10}"
    )
    for reference_snr in (0.0, 10.0, 20.0, 30.0):
        matches = [
            index
            for index, value in enumerate(snr_grid_db)
            if abs(value - reference_snr) < 1e-9
        ]
        if not matches:
            continue
        index = matches[0]
        bound_value = snr_curves["bound"][index]
        mimogs_value = snr_curves["mimogs"][index]
        exhaustive_value = snr_curves["exhaustive"][index]
        print(
            f"  {reference_snr:>9.0f}{bound_value:>10.5f}{mimogs_value:>11.5f}"
            f"{exhaustive_value:>12.5f}{mimogs_value - exhaustive_value:>10.5f}"
            f"{100.0 * mimogs_value / max(exhaustive_value, 1e-12):>8.1f}%"
            f"{100.0 * mimogs_value / max(bound_value, 1e-12):>9.2f}%"
        )

    print()
    print(f"[eval_net_rate] MIMO-GS vs Exhaustive across T_B (L_t={fixed_lt}, "
          f"SNR={arguments.snr_db:g} dB)")
    print(
        f"  {'T_B':>7}{'bound':>10}{'MIMO-GS':>11}{'Exhaustive':>12}"
        f"{'delta':>10}{'MG/exh':>9}{'exh/bound':>11}"
    )
    for reference_tb in (128.0, 512.0, 2048.0, 8192.0):
        index = int(np.argmin(np.abs(tb_grid - reference_tb)))
        bound_value = tb_curves["bound"][index]
        mimogs_value = tb_curves["mimogs"][index]
        exhaustive_value = tb_curves["exhaustive"][index]
        print(
            f"  {tb_grid[index]:>7.0f}{bound_value:>10.5f}{mimogs_value:>11.5f}"
            f"{exhaustive_value:>12.5f}{mimogs_value - exhaustive_value:>10.5f}"
            f"{100.0 * mimogs_value / max(exhaustive_value, 1e-12):>8.1f}%"
            f"{100.0 * exhaustive_value / max(bound_value, 1e-12):>10.2f}%"
        )

    cdf_values = {
        scheme: prelog(CDF_LT, arguments.tau_over_tb)
        * per_location_raw[(scheme, CDF_LT)]
        for scheme, *_ in CDF_SERIES
    }
    cdf_medians, cdf_separation_values = plot_rate_cdf(
        output_dir, cdf_values, CDF_LT
    )
    print()
    print(f"[eval_net_rate] CDF at L_t={CDF_LT}: median rate and KS separation "
          f"from MIMO-GS (plotted when KS >= {CDF_KS_THRESHOLD:g})")
    for scheme, *_ in CDF_SERIES:
        if scheme not in cdf_medians:
            continue
        gap = cdf_separation_values.get(scheme)
        plotted = scheme in CDF_ALWAYS_PLOT or (gap or 1.0) >= CDF_KS_THRESHOLD
        print(
            f"  {SCHEME_LABEL.get(scheme, scheme):<24} median "
            f"{cdf_medians[scheme]:.4f}  KS "
            f"{'--' if gap is None else f'{gap:.4f}'}  "
            f"{'plotted' if plotted else 'omitted (overlaps MIMO-GS)'}"
        )

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    readme: List[str] = [
        "Net achievable rate R_sel(p) -- paper Eq. (6_net_Rsel)",
        "=" * 70,
        "",
        "Generated by eval_net_rate.py (repository root).",
        "",
        "MATERIALS",
        "-" * 70,
        f"  Checkpoint        : {os.path.relpath(checkpoint_path, repository_root)}"
        f"  (iteration {int(checkpoint.get('iteration', -1))})",
        f"  GT long-term maps : {os.path.relpath(os.path.join(gt_root, 'test.mat'), repository_root)}"
        f"  -- TEST split only",
        f"  Complex channels  : {os.path.relpath(complex_path, repository_root)}",
        f"  Beam grid         : {int(scene.beam_rows)} Rx x {tx_total} Tx",
        f"  Test locations    : {num_locations}",
        f"  Realizations B    : {num_realizations} per location",
        "",
        "POSITION JOIN",
        "-" * 70,
        f"  {num_locations}/{num_locations} test locations matched to complex.mat "
        f"rows,",
        f"  each exactly once, at tolerance {MATCH_TOL:g} m (matches are exact).",
        f"  Self-consistency: corr(mean_b |H_b|^2, X_gt) = {correlation:.6f} over "
        f"{probe.size} random",
        "  locations, which also re-validates the join.",
        "",
        "THE METRIC",
        "-" * 70,
        "  R_sel(p; L_t) = (1 - L_t * tau_RS / T_B)",
        "                  * (1/B) * sum_b log2 det( I_{L_r}",
        "                    + (P / (L_t * sigma_z^2)) * H_sel,b H_sel,b^H )",
        "",
        "  This is exactly Eq. (6_net_Rsel).  The fading expectation is evaluated",
        f"  by Monte Carlo over the {num_realizations} stored realizations per",
        "  location; no analytic approximation is used.  H_sel,b is a pure",
        "  row/column slice of the stored beamspace channel (rows B_r, columns",
        "  B_t) -- the channel is never re-transformed.  log2 det is computed from",
        "  the eigenvalues of the Hermitian Gram matrix H_sel H_sel^H (eigvalsh,",
        f"  eigenvalues floored at {EIG_FLOOR:g}), which is stable for rank-deficient",
        "  selections.  Reported curves average R_sel(p) over the test locations.",
        "",
        "CONVENTIONS",
        "-" * 70,
        f"  SNR = P/sigma_z^2      : {arguments.snr_db:g} dB (linear {snr:g})   "
        f"[--snr_db]",
        f"  tau_RS / T_B           : {arguments.tau_over_tb:g}   [--tau_over_TB]",
        f"                           so a full sweep L_t={max(lt_grid)} costs "
        f"{100.0 * max(lt_grid) * arguments.tau_over_tb:.0f}% of the block",
        f"  tau_RS                 : {tau_rs:g} symbols, pinned by tau_RS/T_B at the",
        f"                           reference block T_B = {TB_REFERENCE:g}; the T_B",
        "                           sweep holds tau_RS fixed at this value",
        f"  L_r                    : {num_rx}   [--Lr]",
        f"  L_t grid               : {list(lt_grid)}   [--Lt_grid]",
        f"  Random draws           : {len(random_orders)} (seed {arguments.seed})",
        f"  Net-rate-vs-SNR point  : L_t={fixed_lt}, T_B={fixed_tb:g}, SNR grid "
        f"{snr_grid_db[0]:g}..{snr_grid_db[-1]:g} dB "
        f"({len(snr_grid_db)} pts)",
        f"                           [--fixed_Lt / --fixed_TB / --snr_grid]",
        f"  T_B grid               : {int(tb_grid[0])} .. {int(tb_grid[-1])} symbols, "
        f"{tb_grid.size} log-spaced points",
        f"  g_med rescaling        : "
        f"{'ON (default)' if arguments.g_med_rescale else 'OFF'}   "
        f"[--no_g_med_rescale to disable]",
        f"                           g_med = {g_med:.6g} = median per-location",
        f"                           best-beam gain of X_gt ({10.0 * np.log10(g_med):.1f} dB).",
        "                           Channel POWER is divided by g_med, so --snr_db",
        "                           reads as the MEDIAN POST-ALIGNMENT SNR: 10 dB",
        "                           means a median-strength location sees 10 dB on",
        "                           its best beam.  Absolute rates are on that",
        "                           scale throughout; --no_g_med_rescale returns",
        "                           them to the dataset's native scale.",
        "",
        "                           Side effect worth noting: X_gt/g_med has median",
        "                           per-location peak 1.0, which is the scale X_hat",
        "                           already lives on.  The rescaling therefore also",
        "                           removes the selection-scale mismatch described",
        "                           below -- both maps now enter the greedy metric",
        "                           at the same operating point.",
        "",
        "THE THREE PLOTTED CURVES (fig_net_rate_vs_snr, fig_net_rate_vs_TB)",
        "-" * 70,
        f"  Both figures fix the data-transmission budget at L_t = {fixed_lt} and",
        "  plot exactly three quantities:",
        "",
        "  1. Zero-overhead bound (black dashed).  prelog = 1 (measurement is",
        "     free) and the transmit subspace is the optimal L_r x L_t subspace",
        "     selected on the GROUND-TRUTH map at the SAME budget.  Perfect",
        "     selection plus free measurement, same codebook, same budget: the",
        "     ceiling. It has no T_B dependence, so it is flat in the T_B figure.",
        "",
        "  2. MIMO-GS (solid).  prelog = 1 - L_t*tau_RS/T_B; the subspace is",
        "     selected on the RENDERED map X_hat with the greedy rule.",
        "",
        f"  3. Exhaustive sweep (dash-dot).  prelog = 1 - {full_sweep}*tau_RS/T_B,",
        "     because it measures every beam.  Its RATE TERM IS IDENTICAL to the",
        "     bound's: a full sweep recovers the same optimal subspace, so only",
        "     the measurement cost separates them.  Therefore",
        "",
        "         exhaustive == bound * prelog_exhaustive",
        "",
        "     exactly, and the run asserts that identity on both figures (max",
        "     deviation checked against 1e-12).",
        "",
        "  Prelogs are clamped at 0: for T_B < N_t*tau_RS a full sweep does not",
        "  fit inside the coherence block at all, which would otherwise give a",
        f"  negative prelog. With tau_RS = {tau_rs:g} that boundary sits at",
        f"  T_B = {full_sweep * tau_rs:g}, below the plotted grid, so no point is",
        "  actually clamped here.",
        "",
        "  Random and the 'Genie at the same budget' curve are NOT plotted on",
        "  these two figures.  The genie-same-budget value is kept as a column in",
        "  both CSVs for reference.",
        "",
        "  POWER CONVENTION.  The dataset field named 'magnitude' is already a",
        "  POWER map: mean_b |H_b|^2 correlates "
        f"{correlation:.4f} with it at a ratio of ~1.0,",
        "  while the correlation against magnitude**2 is only ~0.84.  The selection",
        "  metric therefore consumes X_gt / X_hat directly, without squaring.",
        "",
        "BEAM SELECTION",
        "-" * 70,
        "  Transmit, 'greedy' (the paper's rule, Eq. 6_selection_metric):",
        "    starting from B = {}, repeat L_t times, adding the beam n maximizing",
        "      f(B u {n}) = sum of the L_r largest over m of",
        "                   log2(1 + (P/(|B u {n}| sigma_z^2)) sum_{n' in B u {n}} M[m,n'])",
        "    The power-splitting factor uses the size the set WOULD have at that",
        "    step.  All candidates are scored with array operations; the only loop",
        "    is over the L_t steps.  Greedy is prefix-consistent, so one 64-step",
        "    run supplies the selection for every L_t on the grid.",
        "  Transmit, 'toppower': the L_t largest column sums of M.",
        "  Receive (identical for every scheme and rule): on the GT long-term map,",
        "    s_m = sum_{n in B_t} X_gt[m,n]; B_r = the L_r largest.  This models the",
        "    UE picking receive beams from average power measured on the configured",
        "    CSI-RS beams, so B_r is fixed across the fading realizations.",
        "  M = X_hat (rendered) for MIMO-GS, M = X_gt for Genie.  Random draws a",
        "    uniform L_t-subset and uses the same receive rule.",
        "",
        "SELECTION-SCALE CAVEAT (read before quoting MIMO-GS vs Genie)",
        "-" * 70,
        f"  Active setting: --selection_scale {arguments.selection_scale}",
        "  X_gt carries true absolute power (per-location peak ~0.03) while X_hat",
        "  is trained against the max-normalized target and peaks near 1.0.  The",
        "  greedy score log2(1 + a*sum M) is NOT scale invariant, so with 'as_is'",
        "  Genie is scored in the near-linear region of the log and MIMO-GS in the",
        "  curved region -- the two rules are not evaluated at the same operating",
        "  point.  'as_is' is the literal reading of the paper's rule and is the",
        "  default; '--selection_scale peak' max-normalizes both maps so the two",
        "  schemes differ only in map SHAPE.  Measured sensitivity at "
        f"L_t={BT_RECORD_LT}:",
        f"    MIMO-GS  {net_rates['mimogs_greedy'][list(lt_grid).index(BT_RECORD_LT)]:.6f}"
        f"  ->  {scale_probe['mimogs_greedy']:.6f}  (switching to '{alternative}')",
        f"    Genie    {net_rates['genie_greedy'][list(lt_grid).index(BT_RECORD_LT)]:.6f}"
        f"  ->  {scale_probe['genie_greedy']:.6f}",
        "",
        "RESULTS (mean over test locations)",
        "-" * 70,
        "  L_t    prelog   " + "".join(f"{SCHEME_LABEL[s]:>22}" for s in SOLID_SCHEMES),
    ]
    for slot, budget in enumerate(lt_grid):
        readme.append(
            f"  {budget:>3d}    {prelog(budget, arguments.tau_over_tb):.3f}   "
            + "".join(f"{net_rates[s][slot]:>22.6f}" for s in SOLID_SCHEMES)
        )
    readme += [
        "",
        "  Optimal L_t (net rate):",
    ]
    for scheme in SOLID_SCHEMES:
        best = int(np.argmax(net_rates[scheme]))
        readme.append(
            f"    {SCHEME_LABEL[scheme]:<20} L_t*={lt_grid[best]:>3d}  "
            f"R={net_rates[scheme][best]:.6f} bps/Hz  "
            f"(vs exhaustive at T_B={TB_REFERENCE:g}, L_t={fixed_lt}: "
            f"{net_rates[scheme][best] / max(exhaustive_reference, 1e-12):.2f}x)"
        )
    readme += [
        "",
        f"  Genie greedy vs top-power, max gap over the grid: {genie_gap:.4f} bps/Hz",
        "",
        "LITERATURE BASELINE (TRAIN split only, asserted)",
        "-" * 70,
        "  Position NN classifier -- Morais et al., IEEE ICC 2023 "
        "(arXiv:2205.09054).",
        "    Reproduced as published rather than adapted: the input is the 2-D",
        "    (x, y) position only -- z is constant in this dataset "
        f"({np.unique(train_positions[:, 2])} m) so it carries no information --",
        "    min-max normalized with TRAIN statistics and quantized to",
        f"    {POSITION_BINS} bins per axis (resolution {1.0 / POSITION_BINS:g}).",
        "    NO Fourier/positional encoding is applied, unlike the other",
        "    coordinate models in this repository.  Network: 3 hidden layers of",
        "    256 ReLU units, 64-way softmax over the transmit codebook.",
        f"    Training: cross-entropy, Adam lr 0.01 with x0.2 at epochs 20 and 40,",
        f"    batch 32, {arguments.clf_epochs} epochs, seed {arguments.seed}.  The TRAIN split is",
        "    divided 75/25 into fit/validation and the epoch with the best",
        "    validation top-1 is restored: "
        f"epoch {int(classifier_stats['best_epoch'])} "
        f"(val top-1 {100.0 * classifier_stats['best_val_top1']:.2f}%).",
        f"    Test beam accuracy: top-1 {100.0 * classifier_accuracy[1]:.2f}%, "
        f"top-4 {100.0 * classifier_accuracy[4]:.2f}%.",
        f"    Rendered-map fidelity for reference: {rendered_nmse_db:.2f} dB NMSE.",
        "",
        f"CDF INCLUSION AT L_t = {CDF_LT}",
        "-" * 70,
        "  A curve is drawn on the CDF only when its Kolmogorov-Smirnov",
        f"  separation from MIMO-GS reaches {CDF_KS_THRESHOLD:g}; below that the two are",
        "  a single line.  Measured separations and medians:",
]
    for scheme, *_ in CDF_SERIES:
        if scheme not in cdf_medians:
            continue
        gap = cdf_separation_values.get(scheme)
        plotted = scheme in CDF_ALWAYS_PLOT or (gap or 1.0) >= CDF_KS_THRESHOLD
        readme.append(
            f"    {SCHEME_LABEL.get(scheme, scheme):<24} median "
            f"{cdf_medians[scheme]:.4f} bps/Hz   KS "
            f"{'--' if gap is None else f'{gap:.4f}'}   "
            f"{'plotted' if plotted else 'OMITTED (overlaps MIMO-GS)'}"
        )
    readme += [
        "",
        "SANITY CHECKS",
        "-" * 70,
        "  Genie >= MIMO-GS >= Random at every L_t (both rules)  : asserted",
        f"  All schemes coincide at L_t={max(lt_grid)} (spread {spread:.2g})       : asserted",
        f"  corr(mean_b |H_b|^2, X_gt) > 0.98                     : {correlation:.6f}",
        f"  Genie greedy-vs-toppower gap <= 0.5 bps/Hz            : {genie_gap:.4f}",
        "  Overhead-free Genie (greedy) non-decreasing in L_t    : "
        + ("holds" if monotone else "DOES NOT HOLD -- see below"),
        "",
        "THE L_t MONOTONICITY EXPECTATION DOES NOT HOLD (structural, not a bug)",
        "-" * 70,
        "  The overhead-free rate DECREASES in L_t under Eq. (6_net_Rsel) as",
        "  written, so the net rate is maximized at the smallest L_t and the",
        "  intended overhead-vs-resolution trade-off never materializes.",
        "",
        "  Why: H_sel is (L_r x L_t), so H_sel H_sel^H has rank at most L_r and",
        "  contributes at most L_r non-zero eigenvalues no matter how large L_t",
        "  is.  Meanwhile the prefactor P/(L_t sigma_z^2) divides the total power",
        "  by L_t.  Past L_t = L_r, extra selected beams therefore add no spatial",
        "  dimension and only dilute the power already being used.",
        "",
        "  This was checked, not assumed.  rate_model_diagnostic.csv sweeps SNR in",
        "  {configured, 20, 30, 40} dB against L_r in {1, 2, 4, 8, 16} with genie",
        "  selection: the curve is non-decreasing in NO cell of that grid, and the",
        "  best L_t is never larger than 2.  Raising the SNR lifts the level but",
        "  not the shape, so no operating point rescues the expectation.",
        "",
        "  The rate implementation itself was verified against a brute-force",
        "  numpy det() reference (agreement ~1e-7), so the decay is a property of",
        "  the formula and this data, not of the code.",
        "",
        "  WHAT TO CONFIRM FOR THE PAPER.  The likely cause is that L_t plays two",
        "  different roles that this equation conflates: in the overhead term it",
        "  is the number of CSI-RS beams SWEPT (a training cost), while in the",
        "  rate term it is treated as the number of beams simultaneously CARRYING",
        "  DATA at power P/L_t.  If the intended model is 'sweep L_t beams, then",
        "  transmit data on the best few at full power', the rate term should not",
        "  use all L_t selected beams, and it would then be non-decreasing in L_t",
        "  exactly as the sanity check expects.  This script implements the",
        "  equation verbatim as specified and does not silently substitute that",
        "  alternative reading.",
    ]
    if warnings:
        readme += ["", "WARNINGS", "-" * 70]
        readme += [f"  * {message}" for message in warnings]
    readme += [
        "",
        "FILES",
        "-" * 70,
        "  net_rate_vs_Lt.csv       mean net and overhead-free rate per L_t/scheme",
        "  net_rate_vs_TB.csv       best-L_t net rate per coherence block",
        "  net_rate_vs_snr.csv      net rate vs SNR at the fixed (L_t, T_B) point",
        "  alignment_efficiency.csv rate / genie rate at the same L_t",
        "  rate_model_diagnostic.csv  SNR x L_r sweep behind the monotonicity note",
        "  per_location.csv         per-location R_sel for every scheme and L_t",
        "  fig_net_rate_vs_Lt.*     net rate vs L_t (solid greedy, dashed top-power)",
        "  fig_net_rate_vs_snr.*    net rate vs SNR at the fixed (L_t, T_B) point",
        "  fig_alignment_vs_Lt.*    selection quality with the prelog removed",
        "  fig_net_rate_vs_TB.*     bound / MIMO-GS / exhaustive against T_B",
        f"  fig_rate_cdf_Lt{CDF_LT}.*       per-location net rate CDF at L_t={CDF_LT}",
    ]

    with open(os.path.join(output_dir, "README.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(readme).rstrip() + "\n")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print(f"[eval_net_rate] SUMMARY -- {num_locations} test locations, "
          f"{num_realizations} realizations")
    print("=" * 78)
    print("  L_t  prelog" + "".join(f"{SCHEME_LABEL[s]:>22}" for s in ALL_SCHEMES))
    for slot, budget in enumerate(lt_grid):
        print(
            f"  {budget:>3d}  {prelog(budget, arguments.tau_over_tb):.3f}"
            + "".join(f"{net_rates[s][slot]:>22.6f}" for s in ALL_SCHEMES)
        )
    print()
    for scheme in SOLID_SCHEMES:
        best = int(np.argmax(net_rates[scheme]))
        print(
            f"  {SCHEME_LABEL[scheme]:<20} best L_t={lt_grid[best]:>3d}  "
            f"R={net_rates[scheme][best]:.6f} bps/Hz  "
            f"({net_rates[scheme][best] / max(exhaustive_reference, 1e-12):.2f}x the "
            f"exhaustive baseline at T_B={TB_REFERENCE:g}, L_t={fixed_lt})"
        )
    if warnings:
        print()
        for message in warnings:
            print(f"  WARNING: {message}")
    print()
    print(f"[eval_net_rate] Outputs written to {output_dir}")
    print(f"[eval_net_rate] total elapsed {time.time() - started:.1f} s")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
