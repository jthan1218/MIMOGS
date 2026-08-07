"""Per-location beam-map inspection for the Sionna-vs-MIMO-GS NMSE gap.

Runs with zero arguments::

    python eval_gap_inspection.py

``eval_baseline_rt.py`` produced a spatial gap map with blue corridors (Sionna
RT far better) cutting through red blocks (MIMO-GS better).  The hypothesis
under test here is:

    blue  = LoS-dominated locations whose beam map is a single geometric peak
            that both ray tracers place in the same cell,
    red   = NLoS locations whose peak itself comes from material-dependent
            reflections, which an uncalibrated ray tracer misplaces.

The script does two things.  It renders nine individual beam maps (three deep
blue, three near-zero, three deep red, each group spatially spread out) so the
peak structure can be read directly, and it tests the hypothesis quantitatively
over ALL matched locations by correlating the ground-truth peak concentration
ratio against both the gap and Sionna's own NMSE.

Everything that also exists in ``eval_baseline_rt.py`` -- position matching,
checkpoint loading, the gain-MLP width fix, the rendering path and the metric
code -- is imported from it rather than re-implemented, and the recomputed
per-location NMSEs are asserted against that script's CSV before anything is
reported.

Power convention
----------------
The datasets store ``magnitude`` (an amplitude, |H|); the repository's own
metric code squares it to get power (see ``eval_render.topk_metrics``).  The
beam maps here are therefore plotted as POWER,

    P = magnitude**2,   P_dB = 10*log10(P / P.max()) == 20*log10(|H| / |H|.max())

so the "-40 dB floor" is 40 dB of power dynamic range.  The peak concentration
ratio is likewise computed on power, which is the quantity "LoS dominance"
actually refers to; the amplitude-domain ratio is carried in the CSV as a
secondary column so the choice is visible rather than buried.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

# Reused wholesale from the E2 script so the two cannot drift apart.
from eval_baseline_rt import (
    DEFAULT_CKPT,
    DEFAULT_MATCH_TOL,
    DEFAULT_SIONNA_MAT,
    gain_net_hidden_dim,
    gain_net_width,
    load_raw_mat,
    match_positions,
    render_mimogs,
    save_figure,
    score_prediction,
)
from eval_render import build_scene_and_model, resolve_run_dir, restore_config
from utils.loss import normalize_mag_map


DB_FLOOR = -40.0
MIN_SEPARATION_M = 10.0
PER_GROUP = 3
NUM_PCR_BINS = 10

GROUPS = (
    ("deep_blue", "Deep blue -- Sionna RT wins by the most"),
    ("near_zero", "Near zero -- the two predictors tie"),
    ("deep_red", "Deep red -- MIMO-GS wins by the most"),
)

REFERENCE_CSV = os.path.join("comparison_rt", "metrics_per_location.csv")


# ----------------------------------------------------------------------
# Concentration statistics
# ----------------------------------------------------------------------
def concentration_ratios(power_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(peak_ratio, top4_ratio)`` for ``(B, Nr*Nt)`` power maps.

    ``peak_ratio`` is the PCR: the share of the map's total power that sits in
    its single strongest beam pair.  Both are ratios, so they are invariant to
    any per-location rescaling of the map.
    """
    total = power_flat.sum(axis=1)
    total = np.where(total > 0.0, total, np.finfo(np.float64).tiny)

    peak = power_flat.max(axis=1)
    k = min(4, power_flat.shape[1])
    top4 = np.sort(power_flat, axis=1)[:, -k:].sum(axis=1)

    return peak / total, top4 / total


# ----------------------------------------------------------------------
# Location selection
# ----------------------------------------------------------------------
def select_spread_out(
    order: Sequence[int],
    positions: np.ndarray,
    count: int,
    min_separation: float,
) -> List[int]:
    """Greedily take ``count`` rows from ``order`` that are far enough apart.

    ``order`` is already sorted best-first for the group.  A candidate is taken
    only if it is at least ``min_separation`` metres from every row already
    taken, so a group cannot collapse onto one street corner.  If the
    separation cannot be satisfied the constraint is relaxed rather than
    returning fewer than ``count`` rows, and the caller is told.
    """
    chosen: List[int] = []

    for row in order:
        candidate = positions[row]
        if all(
            float(np.linalg.norm(candidate - positions[taken])) >= min_separation
            for taken in chosen
        ):
            chosen.append(int(row))
            if len(chosen) == count:
                return chosen

    for row in order:
        if len(chosen) == count:
            break
        if int(row) not in chosen:
            chosen.append(int(row))

    return chosen


def select_groups(
    gap_db: np.ndarray, positions: np.ndarray
) -> Dict[str, List[int]]:
    """Deep-blue / near-zero / deep-red rows, spatially spread within each."""
    ascending = np.argsort(gap_db, kind="stable")

    orders = {
        "deep_blue": ascending,                              # most negative gap
        "near_zero": np.argsort(np.abs(gap_db), kind="stable"),
        "deep_red": ascending[::-1],                         # most positive gap
    }

    selected: Dict[str, List[int]] = {}
    already: set = set()
    for name, order in orders.items():
        # A row already used by another group would make two figures identical.
        filtered = [int(row) for row in order if int(row) not in already]
        rows = select_spread_out(filtered, positions, PER_GROUP, MIN_SEPARATION_M)
        selected[name] = rows
        already.update(rows)

    return selected


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def to_db(power_map: np.ndarray) -> np.ndarray:
    """Per-panel max-normalized power in dB, floored for display."""
    peak = float(power_map.max())
    if peak <= 0.0:
        return np.full_like(power_map, DB_FLOOR, dtype=np.float64)
    with np.errstate(divide="ignore"):
        db = 10.0 * np.log10(np.maximum(power_map / peak, 1e-30))
    return np.maximum(db, DB_FLOOR)


def plot_location_maps(
    output_dir: str,
    stem: str,
    group_title: str,
    position: np.ndarray,
    gt_power: np.ndarray,
    sionna_power: np.ndarray,
    mimogs_power: np.ndarray,
    sionna_nmse: float,
    mimogs_nmse: float,
    gap: float,
    gt_peak_rc: Tuple[int, int],
) -> None:
    """Ground truth | Sionna RT | MIMO-GS for one location, shared dB scale."""
    panels = (
        ("Ground truth (WI)", gt_power),
        ("Sionna RT", sionna_power),
        ("MIMO-GS", mimogs_power),
    )

    figure, axes = plt.subplots(1, 3, figsize=(14.0, 3.5), layout="constrained")

    image = None
    for axis, (title, power_map) in zip(axes, panels):
        image = axis.imshow(
            to_db(power_map),
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=DB_FLOOR,
            vmax=0.0,
        )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Tx beam index", fontsize=9)
        axis.tick_params(labelsize=8)

        # Same cell on every panel: the GT peak, so misplacement is obvious.
        peak_row, peak_col = gt_peak_rc
        axis.add_patch(
            Rectangle(
                (peak_col - 0.5, peak_row - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="red",
                linewidth=1.3,
            )
        )

    axes[0].set_ylabel("Rx beam index", fontsize=9)

    colorbar = figure.colorbar(image, ax=axes.tolist(), fraction=0.02, pad=0.01)
    colorbar.set_label("Normalized power [dB]")

    figure.suptitle(
        f"{group_title}\n"
        f"(x, y) = ({position[0]:.1f}, {position[1]:.1f}) m   |   "
        f"NMSE Sionna {sionna_nmse:.2f} dB   "
        f"MIMO-GS {mimogs_nmse:.2f} dB   "
        f"gap {gap:+.2f} dB   |   red square = GT argmax",
        fontsize=10,
    )

    figure.savefig(os.path.join(output_dir, f"{stem}.png"), dpi=200)
    figure.savefig(os.path.join(output_dir, f"{stem}.pdf"))
    plt.close(figure)


def binned_median(
    x_values: np.ndarray, y_values: np.ndarray, num_bins: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Median of ``y`` inside ``num_bins`` equal-count bins of ``x``."""
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.unique(np.quantile(x_values, quantiles))
    if edges.size < 2:
        return np.empty(0), np.empty(0)

    centers: List[float] = []
    medians: List[float] = []
    for index in range(edges.size - 1):
        low, high = edges[index], edges[index + 1]
        if index == edges.size - 2:
            mask = (x_values >= low) & (x_values <= high)
        else:
            mask = (x_values >= low) & (x_values < high)
        if not np.any(mask):
            continue
        centers.append(float(np.median(x_values[mask])))
        medians.append(float(np.median(y_values[mask])))

    return np.asarray(centers), np.asarray(medians)


def plot_pcr_scatter(
    output_dir: str,
    stem: str,
    pcr: np.ndarray,
    y_values: np.ndarray,
    y_label: str,
    title: str,
    zero_line: bool,
) -> float:
    """Scatter PCR vs. ``y`` with a binned-median overlay; return Spearman rho."""
    rho, p_value = spearmanr(pcr, y_values)

    figure, axis = plt.subplots(figsize=(7.4, 5.2))
    axis.scatter(pcr, y_values, s=5, alpha=0.18, color="tab:blue", linewidths=0.0)

    centers, medians = binned_median(pcr, y_values, NUM_PCR_BINS)
    if centers.size:
        axis.plot(
            centers,
            medians,
            color="tab:red",
            marker="o",
            markersize=5,
            linewidth=2.0,
            label=f"binned median ({centers.size} quantile bins)",
        )

    if zero_line:
        axis.axhline(0.0, color="0.35", linewidth=1.0, linestyle="--", zorder=1)

    axis.set_xscale("log")
    axis.set_xlabel("GT peak concentration ratio  max(P) / sum(P)   [log scale]")
    axis.set_ylabel(y_label)
    axis.set_title(title, fontsize=11)
    axis.grid(alpha=0.3, linewidth=0.5)
    axis.legend(fontsize=8, loc="best")

    axis.text(
        0.02,
        0.03,
        f"Spearman rho = {rho:+.3f}\np = {p_value:.2e}\nN = {pcr.size}",
        transform=axis.transAxes,
        fontsize=9,
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "lw": 0.5},
    )

    save_figure(figure, output_dir, stem)
    return float(rho)


# ----------------------------------------------------------------------
# Cross-check against eval_baseline_rt.py
# ----------------------------------------------------------------------
def crosscheck_against_reference(
    reference_path: str,
    gt_indices: np.ndarray,
    mimogs_nmse: np.ndarray,
    sionna_nmse: np.ndarray,
    sample_size: int,
    tolerance: float,
) -> Optional[float]:
    """Assert the recomputed NMSEs match the E2 CSV on a random sample."""
    if not os.path.isfile(reference_path):
        print(
            f"[eval_gap_inspection] WARNING: reference CSV '{reference_path}' is "
            f"missing, so the cross-check against eval_baseline_rt.py was skipped. "
            f"Run 'python eval_baseline_rt.py --allow_partial_match' to create it."
        )
        return None

    reference: Dict[int, Tuple[float, float]] = {}
    with open(reference_path, "r", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            reference[int(record["gt_test_index"])] = (
                float(record["mimogs_NMSE_shape_dB"]),
                float(record["sionna_NMSE_shape_dB"]),
            )

    lookup = {int(value): row for row, value in enumerate(gt_indices)}
    shared = sorted(set(lookup) & set(reference))
    if not shared:
        raise SystemExit(
            "[eval_gap_inspection] The reference CSV shares no gt_test_index with "
            "the current matching; the two scripts are not looking at the same set."
        )

    generator = np.random.default_rng(0)
    sample = generator.choice(
        np.asarray(shared, dtype=np.int64),
        size=min(sample_size, len(shared)),
        replace=False,
    )

    worst = 0.0
    for gt_index in sample:
        row = lookup[int(gt_index)]
        expected_mimogs, expected_sionna = reference[int(gt_index)]
        worst = max(
            worst,
            abs(float(mimogs_nmse[row]) - expected_mimogs),
            abs(float(sionna_nmse[row]) - expected_sionna),
        )

    assert worst <= tolerance, (
        f"Recomputed NMSE differs from {reference_path} by up to {worst:.6g} dB "
        f"(tolerance {tolerance:g} dB); the metric plumbing diverged from "
        f"eval_baseline_rt.py."
    )
    print(
        f"[eval_gap_inspection] cross-check vs. eval_baseline_rt.py: max |delta| = "
        f"{worst:.3g} dB over {sample.size} sampled locations (tol {tolerance:g}) -- OK"
    )
    return worst


# ----------------------------------------------------------------------
# CSV writers
# ----------------------------------------------------------------------
def write_inspection_summary(path: str, records: Sequence[Dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_pcr_csv(
    path: str,
    gt_indices: np.ndarray,
    positions: np.ndarray,
    pcr: np.ndarray,
    pcr_amplitude: np.ndarray,
    top4: np.ndarray,
    mimogs_nmse: np.ndarray,
    sionna_nmse: np.ndarray,
    gap_db: np.ndarray,
) -> None:
    header = [
        "gt_test_index",
        "x",
        "y",
        "z",
        "pcr_power",
        "top4_ratio_power",
        "pcr_amplitude",
        "mimogs_NMSE_shape_dB",
        "sionna_NMSE_shape_dB",
        "nmse_gap_dB",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in range(positions.shape[0]):
            writer.writerow(
                [
                    int(gt_indices[row]),
                    f"{positions[row, 0]:.6f}",
                    f"{positions[row, 1]:.6f}",
                    f"{positions[row, 2]:.6f}",
                    f"{pcr[row]:.9g}",
                    f"{top4[row]:.9g}",
                    f"{pcr_amplitude[row]:.9g}",
                    f"{mimogs_nmse[row]:.6f}",
                    f"{sionna_nmse[row]:.6f}",
                    f"{gap_db[row]:.6f}",
                ]
            )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-location beam-map inspection of the Sionna/MIMO-GS NMSE gap"
    )
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--sionna_mat", type=str, default=DEFAULT_SIONNA_MAT)
    parser.add_argument("--match_tol", type=float, default=DEFAULT_MATCH_TOL)
    parser.add_argument("--outputs_root", type=str, default="outputs")
    parser.add_argument("--analysis_root", type=str, default="analysis")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--source_path", type=str, default="")
    parser.add_argument(
        "--crosscheck_samples",
        type=int,
        default=100,
        help="Locations sampled when comparing against eval_baseline_rt.py's CSV.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    repository_root = os.path.dirname(os.path.abspath(__file__))

    outputs_root = arguments.outputs_root
    if not os.path.isabs(outputs_root):
        outputs_root = os.path.join(repository_root, outputs_root)

    run_dir, checkpoint_path = resolve_run_dir(arguments.ckpt, outputs_root)
    run_name = os.path.basename(os.path.normpath(run_dir))

    print("=" * 78)
    print(f"[eval_gap_inspection] RUN        : {run_name}")
    print(f"[eval_gap_inspection] checkpoint : {checkpoint_path}")
    print("=" * 78)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_params, opt_params = restore_config(run_dir, checkpoint)

    if arguments.source_path:
        model_params.source_path = os.path.abspath(arguments.source_path)
    gt_root = str(getattr(model_params, "source_path", ""))
    if not os.path.isdir(gt_root):
        raise SystemExit(
            f"[eval_gap_inspection] Ground-truth dataset '{gt_root}' is missing."
        )

    sionna_mat = arguments.sionna_mat
    if not os.path.isabs(sionna_mat):
        sionna_mat = os.path.join(repository_root, sionna_mat)

    # ------------------------------------------------------------------
    # Match GT test locations onto Sionna locations (raw coordinates)
    # ------------------------------------------------------------------
    gt_test_positions, gt_test_magnitude = load_raw_mat(
        os.path.join(gt_root, "test.mat")
    )
    sionna_positions, sionna_magnitude = load_raw_mat(sionna_mat)

    matched_gt, matched_sionna, _ = match_positions(
        gt_test_positions, sionna_positions, float(arguments.match_tol)
    )
    num_matched = int(matched_gt.size)
    if num_matched == 0:
        raise SystemExit("[eval_gap_inspection] No location matched.")

    matched_positions = gt_test_positions[matched_gt]
    print(
        f"[eval_gap_inspection] matched {num_matched} of "
        f"{gt_test_positions.shape[0]} GT test locations "
        f"(tolerance {arguments.match_tol:g} m)"
    )

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda_rasterizer = (
        bool(int(getattr(model_params, "use_cuda_rasterizer", 1)))
        and device.type == "cuda"
    )
    batch_size = max(
        1, int(arguments.batch_size) or int(getattr(model_params, "batch_size", 8))
    )

    hidden_dim = gain_net_hidden_dim(checkpoint)
    if hidden_dim is not None:
        print(
            f"[eval_gap_inspection] checkpoint gain MLP is {hidden_dim}-wide; "
            f"rebuilding it to match."
        )
    with gain_net_width(hidden_dim):
        scene, gaussians = build_scene_and_model(
            model_params, opt_params, checkpoint, device
        )

    ground_truth = torch.from_numpy(gt_test_magnitude[matched_gt]).to(device)
    target_normalized = normalize_mag_map(ground_truth)
    sionna_prediction = torch.from_numpy(sionna_magnitude[matched_sionna]).to(device)

    print(f"[eval_gap_inspection] rendering {num_matched} MIMO-GS locations ...")
    mimogs_prediction = render_mimogs(
        scene,
        gaussians,
        model_params,
        device,
        scene.test_set.positions[torch.as_tensor(matched_gt)],
        batch_size,
        use_cuda_rasterizer,
    )

    mimogs_scores = score_prediction(mimogs_prediction, target_normalized)
    sionna_scores = score_prediction(sionna_prediction, target_normalized)

    mimogs_nmse = mimogs_scores["nmse_shape_db"]
    sionna_nmse = sionna_scores["nmse_shape_db"]
    gap_db = sionna_nmse - mimogs_nmse

    output_dir = os.path.join(
        repository_root, arguments.analysis_root, run_name, "gap_inspection"
    )
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Sanity: identical numbers to eval_baseline_rt.py
    # ------------------------------------------------------------------
    reference_path = os.path.join(
        repository_root, arguments.analysis_root, run_name, REFERENCE_CSV
    )
    crosscheck_against_reference(
        reference_path,
        matched_gt,
        mimogs_nmse,
        sionna_nmse,
        int(arguments.crosscheck_samples),
        1e-3,
    )

    # ------------------------------------------------------------------
    # Concentration statistics over every matched location
    # ------------------------------------------------------------------
    beam_rows, beam_cols = int(scene.beam_rows), int(scene.beam_cols)

    gt_magnitude_np = gt_test_magnitude[matched_gt].astype(np.float64)
    gt_power = gt_magnitude_np ** 2
    gt_power_flat = gt_power.reshape(num_matched, -1)

    pcr, top4 = concentration_ratios(gt_power_flat)
    pcr_amplitude, _ = concentration_ratios(
        gt_magnitude_np.reshape(num_matched, -1)
    )

    sionna_power = (
        sionna_prediction.detach().cpu().numpy().astype(np.float64) ** 2
    )
    mimogs_power = (
        mimogs_prediction.detach().cpu().numpy().astype(np.float64) ** 2
    )

    gt_argmax = gt_power_flat.argmax(axis=1)
    sionna_argmax = sionna_power.reshape(num_matched, -1).argmax(axis=1)
    mimogs_argmax = mimogs_power.reshape(num_matched, -1).argmax(axis=1)

    write_pcr_csv(
        os.path.join(output_dir, "pcr_all_locations.csv"),
        matched_gt,
        matched_positions,
        pcr,
        pcr_amplitude,
        top4,
        mimogs_nmse,
        sionna_nmse,
        gap_db,
    )

    # ------------------------------------------------------------------
    # Hypothesis figures
    # ------------------------------------------------------------------
    rho_gap = plot_pcr_scatter(
        output_dir,
        "fig_pcr_vs_gap",
        pcr,
        gap_db,
        "NMSE gap [dB]  (Sionna - MIMO-GS; >0 = Sionna worse)",
        "GT peak concentration vs. the NMSE gap",
        zero_line=True,
    )
    rho_sionna = plot_pcr_scatter(
        output_dir,
        "fig_pcr_vs_sionna_nmse",
        pcr,
        sionna_nmse,
        "Sionna RT NMSE [dB]",
        "GT peak concentration vs. Sionna RT accuracy",
        zero_line=False,
    )
    rho_mimogs, _ = spearmanr(pcr, mimogs_nmse)

    print()
    print(f"[eval_gap_inspection] Spearman rho (PCR vs. gap)         = {rho_gap:+.3f}")
    print(f"[eval_gap_inspection] Spearman rho (PCR vs. Sionna NMSE) = {rho_sionna:+.3f}")
    print(f"[eval_gap_inspection] Spearman rho (PCR vs. MIMO-GS NMSE)= {rho_mimogs:+.3f}")

    # ------------------------------------------------------------------
    # Nine inspected locations
    # ------------------------------------------------------------------
    selected = select_groups(gap_db, matched_positions)
    records: List[Dict[str, object]] = []

    print()
    print("[eval_gap_inspection] inspected locations")
    for group_name, group_title in GROUPS:
        rows = selected[group_name]
        separations = [
            float(np.linalg.norm(matched_positions[a] - matched_positions[b]))
            for index, a in enumerate(rows)
            for b in rows[index + 1 :]
        ]
        if separations and min(separations) < MIN_SEPARATION_M - 1e-6:
            print(
                f"  NOTE: group '{group_name}' could not keep "
                f"{MIN_SEPARATION_M:.0f} m separation (min "
                f"{min(separations):.1f} m); the constraint was relaxed."
            )

        for order, row in enumerate(rows, start=1):
            peak_row = int(gt_argmax[row] // beam_cols)
            peak_col = int(gt_argmax[row] % beam_cols)

            stem = f"fig_maps_{group_name}_{order}"
            plot_location_maps(
                output_dir,
                stem,
                f"{group_title}  [{order}/{PER_GROUP}]",
                matched_positions[row],
                gt_power[row],
                sionna_power[row],
                mimogs_power[row],
                float(sionna_nmse[row]),
                float(mimogs_nmse[row]),
                float(gap_db[row]),
                (peak_row, peak_col),
            )

            sionna_hit = int(sionna_argmax[row] == gt_argmax[row])
            mimogs_hit = int(mimogs_argmax[row] == gt_argmax[row])

            records.append(
                {
                    "group": group_name,
                    "rank_in_group": order,
                    "figure": f"{stem}.png",
                    "gt_test_index": int(matched_gt[row]),
                    "x": f"{matched_positions[row, 0]:.6f}",
                    "y": f"{matched_positions[row, 1]:.6f}",
                    "z": f"{matched_positions[row, 2]:.6f}",
                    "sionna_NMSE_shape_dB": f"{sionna_nmse[row]:.6f}",
                    "mimogs_NMSE_shape_dB": f"{mimogs_nmse[row]:.6f}",
                    "nmse_gap_dB": f"{gap_db[row]:.6f}",
                    "pcr_power": f"{pcr[row]:.9g}",
                    "top4_ratio_power": f"{top4[row]:.9g}",
                    "gt_argmax_index": int(gt_argmax[row]),
                    "gt_argmax_rx": peak_row,
                    "gt_argmax_tx": peak_col,
                    "sionna_argmax_index": int(sionna_argmax[row]),
                    "mimogs_argmax_index": int(mimogs_argmax[row]),
                    "sionna_argmax_matches_gt": sionna_hit,
                    "mimogs_argmax_matches_gt": mimogs_hit,
                }
            )

            print(
                f"  {group_name:<10} #{order}  (x,y)=({matched_positions[row, 0]:7.2f},"
                f"{matched_positions[row, 1]:7.2f})  gap {gap_db[row]:+7.2f} dB  "
                f"PCR {pcr[row]:.4f}  top4 {top4[row]:.4f}  "
                f"argmax hit: Sionna {sionna_hit} / MIMO-GS {mimogs_hit}"
            )

    write_inspection_summary(
        os.path.join(output_dir, "inspection_summary.csv"), records
    )

    # ------------------------------------------------------------------
    # Group-level argmax agreement over ALL locations in each gap regime
    # ------------------------------------------------------------------
    sionna_hits_all = (sionna_argmax == gt_argmax).astype(np.float64)
    mimogs_hits_all = (mimogs_argmax == gt_argmax).astype(np.float64)

    tertiles = np.quantile(gap_db, [1.0 / 3.0, 2.0 / 3.0])
    regimes = (
        ("gap < p33 (blue)", gap_db < tertiles[0]),
        ("p33 <= gap < p66", (gap_db >= tertiles[0]) & (gap_db < tertiles[1])),
        ("gap >= p66 (red)", gap_db >= tertiles[1]),
    )

    print()
    print("[eval_gap_inspection] argmax agreement and concentration by gap regime")
    print(
        f"  {'regime':<20}{'N':>7}{'PCR med':>10}{'top4 med':>10}"
        f"{'Sionna hit':>12}{'MIMO-GS hit':>13}{'Sionna NMSE':>13}"
    )
    regime_rows: List[Dict[str, object]] = []
    for label, mask in regimes:
        count = int(mask.sum())
        if count == 0:
            continue
        row = {
            "regime": label,
            "N": count,
            "pcr_median": float(np.median(pcr[mask])),
            "top4_median": float(np.median(top4[mask])),
            "sionna_argmax_hit_rate": float(np.mean(sionna_hits_all[mask])),
            "mimogs_argmax_hit_rate": float(np.mean(mimogs_hits_all[mask])),
            "sionna_NMSE_median_dB": float(np.median(sionna_nmse[mask])),
            "mimogs_NMSE_median_dB": float(np.median(mimogs_nmse[mask])),
        }
        regime_rows.append(row)
        print(
            f"  {label:<20}{count:>7d}{row['pcr_median']:>10.4f}"
            f"{row['top4_median']:>10.4f}{row['sionna_argmax_hit_rate']:>12.4f}"
            f"{row['mimogs_argmax_hit_rate']:>13.4f}"
            f"{row['sionna_NMSE_median_dB']:>13.2f}"
        )

    with open(
        os.path.join(output_dir, "gap_regime_stats.csv"), "w", newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(regime_rows[0].keys()))
        writer.writeheader()
        for row in regime_rows:
            writer.writerow(row)

    # ------------------------------------------------------------------
    # Which candidate explains the gap better: how concentrated the GT map is,
    # or whether Sionna puts the peak in the right cell?  Both are scored on
    # the same target so the comparison is direct.
    # ------------------------------------------------------------------
    rho_hit_gap, p_hit_gap = spearmanr(sionna_hits_all, gap_db)
    hit_mask = sionna_hits_all > 0.5

    candidates = [
        {
            "predictor_of_gap": "GT peak concentration ratio (PCR)",
            "spearman_rho_vs_gap": float(rho_gap),
            "spearman_rho_vs_sionna_nmse": float(rho_sionna),
            "abs_rho_vs_gap": abs(float(rho_gap)),
        },
        {
            "predictor_of_gap": "Sionna argmax == GT argmax (0/1)",
            "spearman_rho_vs_gap": float(rho_hit_gap),
            "spearman_rho_vs_sionna_nmse": float(spearmanr(sionna_hits_all, sionna_nmse)[0]),
            "abs_rho_vs_gap": abs(float(rho_hit_gap)),
        },
    ]

    print()
    print("[eval_gap_inspection] what explains the gap?")
    for candidate in candidates:
        print(
            f"  {candidate['predictor_of_gap']:<36} "
            f"rho vs gap = {candidate['spearman_rho_vs_gap']:+.3f}   "
            f"rho vs Sionna NMSE = {candidate['spearman_rho_vs_sionna_nmse']:+.3f}"
        )
    print(
        f"  Sionna argmax HIT  (N={int(hit_mask.sum())}): median gap "
        f"{np.median(gap_db[hit_mask]):+.2f} dB, median Sionna NMSE "
        f"{np.median(sionna_nmse[hit_mask]):.2f} dB, median PCR "
        f"{np.median(pcr[hit_mask]):.4f}"
    )
    print(
        f"  Sionna argmax MISS (N={int((~hit_mask).sum())}): median gap "
        f"{np.median(gap_db[~hit_mask]):+.2f} dB, median Sionna NMSE "
        f"{np.median(sionna_nmse[~hit_mask]):.2f} dB, median PCR "
        f"{np.median(pcr[~hit_mask]):.4f}"
    )

    with open(
        os.path.join(output_dir, "hypothesis_test.csv"), "w", newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "predictor_of_gap",
                "spearman_rho_vs_gap",
                "spearman_rho_vs_sionna_nmse",
                "abs_rho_vs_gap",
            ],
        )
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(candidate)
        handle.write(
            f"# Sionna argmax HIT  N={int(hit_mask.sum())} "
            f"median_gap_dB={np.median(gap_db[hit_mask]):.4f} "
            f"median_sionna_nmse_dB={np.median(sionna_nmse[hit_mask]):.4f} "
            f"median_pcr={np.median(pcr[hit_mask]):.4f}\n"
        )
        handle.write(
            f"# Sionna argmax MISS N={int((~hit_mask).sum())} "
            f"median_gap_dB={np.median(gap_db[~hit_mask]):.4f} "
            f"median_sionna_nmse_dB={np.median(sionna_nmse[~hit_mask]):.4f} "
            f"median_pcr={np.median(pcr[~hit_mask]):.4f}\n"
        )

    print()
    print(f"[eval_gap_inspection] Outputs written to {output_dir}")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
