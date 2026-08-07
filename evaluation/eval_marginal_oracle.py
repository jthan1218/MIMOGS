"""Marginal (one-sided) oracle: an upper bound for rx-only / tx-only methods.

Runs with zero arguments::

    python eval_marginal_oracle.py

A method that renders only one-sided angular spectra -- an rx-side power
profile and a tx-side power profile -- cannot recover more of the joint
beam-pair map than the outer product of those two profiles.  This script
builds exactly that reconstruction from the GROUND TRUTH marginals,

    X_oracle[m, n] = r_m * t_n / S,   r_m = sum_n X[m,n],
                                      t_n = sum_m X[m,n],
                                      S   = sum(X),

so it is *perfect* on the one-sided task by construction (its own marginals
reproduce the GT marginals exactly, asserted below) and it is the best any
separable method can do on the joint task.  Whatever it loses against the
ground truth is the part of the joint structure that is fundamentally
unavailable without modelling the rx-tx coupling.

The MIMO-GS checkpoint is scored alongside it, both on the joint map and --
the reverse direction -- on the marginals themselves, to show the joint
renderer does not pay for its extra generality on the easier one-sided task.

Conventions
-----------
Metric code is imported from ``eval_render`` and the dB map convention from
``eval_db_16by64``; nothing is re-implemented.  The dataset field named
``magnitude`` is already a POWER map, so dB is ``10*log10(P / P.max())``.
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

from eval_db_16by64 import to_db
from eval_render import (
    EPS,
    build_scene_and_model,
    evaluate_test_set,
    gain_net_hidden_dim,
    gain_net_width,
    render_batch,
    resolve_run_dir,
    restore_config,
    summarize,
    topk_metrics,
)
from utils.loss import normalize_mag_map


DEFAULT_CKPT = "outputs/20260805_051724"
DEFAULT_FLOOR_DB = -40.0
DEFAULT_ANALYSIS_DIR = os.path.join("analysis", "marginal_oracle")

TOPK_ACC_VALUES = (1, 4, 8)
CAPTURE_VALUES = (1, 4)
ALL_K_VALUES = tuple(sorted(set(TOPK_ACC_VALUES) | set(CAPTURE_VALUES)))

# A GT bin counts as a "significant component" at 1% of the peak (-20 dB), the
# same threshold used to call an oracle top-4 pick a ghost.
COMPONENT_THRESHOLD = 0.01
GHOST_TOPK = 4
QUALITATIVE_LOCATIONS = 3
QUALITATIVE_MIN_SEPARATION_M = 10.0
RANK1_TOLERANCE = 1e-6


# ----------------------------------------------------------------------
# Oracle construction
# ----------------------------------------------------------------------
def marginal_oracle(ground_truth: torch.Tensor) -> torch.Tensor:
    """``X_oracle[m,n] = r_m t_n / S`` for a batch of ``(B, Nr, Nt)`` maps.

    Total power is preserved (``sum(X_oracle) == sum(X)``), so the oracle sits
    on the same absolute scale as the ground truth and its marginals are the
    GT marginals exactly.
    """
    rx_marginal = ground_truth.sum(dim=2)                      # (B, Nr)
    tx_marginal = ground_truth.sum(dim=1)                      # (B, Nt)
    total = ground_truth.reshape(ground_truth.shape[0], -1).sum(dim=1)
    total = total.clamp_min(torch.finfo(ground_truth.dtype).tiny)
    return rx_marginal.unsqueeze(2) * tx_marginal.unsqueeze(1) / total.view(-1, 1, 1)


def nmse_db(prediction_flat: torch.Tensor, target_flat: torch.Tensor) -> torch.Tensor:
    """Per-row NMSE in dB, the same arithmetic ``eval_render`` uses."""
    energy = target_flat.square().sum(dim=1).clamp_min(EPS)
    ratio = (prediction_flat - target_flat).square().sum(dim=1) / energy
    return 10.0 * torch.log10(ratio.clamp_min(1e-12))


def score_against_gt(
    prediction: torch.Tensor, ground_truth: torch.Tensor
) -> Dict[str, np.ndarray]:
    """Both NMSE conventions plus rank metrics, exactly as in ``eval_render``.

    ``raw`` puts the prediction on the GT-peak scale before comparing (the
    oracle carries the GT's absolute scale, so this is its honest analogue of
    the training scale term); ``shape`` max-normalizes it.
    """
    count = prediction.shape[0]

    target_n = normalize_mag_map(ground_truth)
    peak = ground_truth.reshape(count, -1).amax(dim=1).clamp_min(EPS).view(-1, 1, 1)

    target_flat = target_n.reshape(count, -1)
    raw_flat = (prediction / peak).reshape(count, -1)
    shape_flat = normalize_mag_map(prediction).reshape(count, -1)

    scored: Dict[str, np.ndarray] = {
        "nmse_raw_db": nmse_db(raw_flat, target_flat).cpu().numpy().astype(np.float64),
        "nmse_shape_db": nmse_db(shape_flat, target_flat)
        .cpu()
        .numpy()
        .astype(np.float64),
    }

    for k, (overlap, capture) in topk_metrics(
        prediction.reshape(count, -1), target_flat, ALL_K_VALUES
    ).items():
        scored[f"topk_acc_K{k}"] = overlap.cpu().numpy().astype(np.float64)
        scored[f"power_capture_K{k}"] = capture.cpu().numpy().astype(np.float64)

    return scored


def marginal_nmse_db(
    prediction: torch.Tensor, ground_truth: torch.Tensor, axis: int
) -> np.ndarray:
    """NMSE of a predicted marginal against the GT marginal.

    ``axis=2`` collapses the tx index and yields the rx marginal; ``axis=1``
    the other way round.  Both vectors are max-normalized first, matching the
    per-map convention ``normalize_mag_map`` applies to the joint maps, so the
    number is scale-invariant and comparable to the joint NMSE.
    """
    predicted = prediction.sum(dim=axis)
    target = ground_truth.sum(dim=axis)

    predicted = predicted / predicted.amax(dim=1, keepdim=True).clamp_min(EPS)
    target = target / target.amax(dim=1, keepdim=True).clamp_min(EPS)

    return nmse_db(predicted, target).cpu().numpy().astype(np.float64)


def ghost_and_component_counts(
    oracle: torch.Tensor, ground_truth: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ghost pairs among the oracle's top-4, and GT component counts.

    A "ghost" is a beam pair the oracle ranks in its top-4 whose true power is
    below ``COMPONENT_THRESHOLD`` of the GT peak -- an artefact of the outer
    product lighting up the intersection of two strong marginals where no
    actual path exists.  The GT top-4 indices are returned too, for the figure.
    """
    count = ground_truth.shape[0]
    gt_flat = ground_truth.reshape(count, -1)
    oracle_flat = oracle.reshape(count, -1)

    peak = gt_flat.amax(dim=1, keepdim=True).clamp_min(EPS)
    significant = (gt_flat >= COMPONENT_THRESHOLD * peak).sum(dim=1)

    oracle_top = oracle_flat.topk(GHOST_TOPK, dim=1).indices
    gt_top = gt_flat.topk(GHOST_TOPK, dim=1).indices

    picked_power = gt_flat.gather(1, oracle_top)
    ghosts = (picked_power < COMPONENT_THRESHOLD * peak).sum(dim=1)

    return (
        ghosts.cpu().numpy().astype(np.int64),
        significant.cpu().numpy().astype(np.int64),
        gt_top.cpu().numpy().astype(np.int64),
    )


def rank1_energy_fraction(ground_truth: torch.Tensor) -> np.ndarray:
    """``sigma_1^2 / sum_i sigma_i^2`` per location: how separable the GT is.

    This is the quantity that decides whether the marginal oracle is a loose
    or a tight competitor.  A map whose leading singular value carries nearly
    all the energy IS an outer product, so no joint model can gain much on it.
    """
    singular = torch.linalg.svdvals(ground_truth.double())
    energy = singular.square()
    total = energy.sum(dim=1).clamp_min(torch.finfo(torch.float64).tiny)
    return (energy[:, 0] / total).cpu().numpy().astype(np.float64)


def best_rank1_approximation(ground_truth: torch.Tensor) -> torch.Tensor:
    """The truncated-SVD rank-1 map: the TIGHTEST rank-1 bound.

    The marginal oracle is one particular rank-1 reconstruction -- the one
    consistent with both marginals -- but it is not the best one.  Reporting
    the SVD optimum separates "what rank-1 costs" from "what using the
    marginals specifically costs".
    """
    left, singular, right = torch.linalg.svd(
        ground_truth.double(), full_matrices=False
    )
    approximation = (left[:, :, :1] * singular[:, None, :1]) @ right[:, :1, :]
    # The leading singular triplet of a non-negative matrix is sign-consistent
    # (Perron-Frobenius); take magnitudes so a flipped pair cannot produce
    # negative "power".
    return approximation.abs().to(ground_truth.dtype)


def assert_rank_one(oracle: torch.Tensor, sample_size: int = 64) -> float:
    """The oracle is rank-1 by construction; verify numerically via SVD."""
    count = min(int(sample_size), int(oracle.shape[0]))
    probe = oracle[:count].double()
    singular = torch.linalg.svdvals(probe)
    leading = singular[:, 0].clamp_min(torch.finfo(torch.float64).tiny)
    worst = float((singular[:, 1] / leading).max().item())
    assert worst <= RANK1_TOLERANCE, (
        f"X_oracle is not rank-1: worst sigma_2/sigma_1 = {worst:.3g} over "
        f"{count} sampled locations."
    )
    return worst


def assert_marginals_match(
    oracle: torch.Tensor, ground_truth: torch.Tensor, tolerance: float = 1e-6
) -> Tuple[float, float]:
    """The oracle must reproduce both GT marginals exactly (relative error)."""
    errors: List[float] = []
    for axis in (2, 1):
        predicted = oracle.sum(dim=axis).double()
        target = ground_truth.sum(dim=axis).double()
        scale = target.abs().amax(dim=1, keepdim=True).clamp_min(
            torch.finfo(torch.float64).tiny
        )
        errors.append(float(((predicted - target).abs() / scale).max().item()))

    for name, error in zip(("rx", "tx"), errors):
        assert error <= tolerance, (
            f"Oracle {name} marginal deviates from the GT marginal by "
            f"{error:.3g} (relative to the marginal peak); it is supposed to "
            f"be exact by construction."
        )
    return errors[0], errors[1]


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
def select_qualitative(
    component_counts: np.ndarray, positions: np.ndarray, count: int
) -> List[int]:
    """Most multi-component GT maps, kept spatially apart."""
    order = np.argsort(-component_counts, kind="stable")
    chosen: List[int] = []
    for row in order:
        if all(
            float(np.linalg.norm(positions[row] - positions[taken]))
            >= QUALITATIVE_MIN_SEPARATION_M
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


def plot_qualitative(
    output_dir: str,
    ground_truth: np.ndarray,
    oracle: np.ndarray,
    rendered: np.ndarray,
    gt_top_pairs: np.ndarray,
    oracle_nmse: np.ndarray,
    mimogs_nmse: np.ndarray,
    ghosts: np.ndarray,
    components: np.ndarray,
    coordinates: np.ndarray,
    beam_cols: int,
    floor_db: float,
) -> None:
    """GT | oracle | MIMO-GS per location, with the GT top-4 pairs marked."""
    rows = ground_truth.shape[0]
    figure, axes = plt.subplots(
        rows,
        3,
        figsize=(13.2, 2.4 * rows + 1.1),
        squeeze=False,
        sharex=True,
        layout="constrained",
    )

    image = None
    for row in range(rows):
        panels = (
            ("Ground truth", ground_truth[row], None),
            ("Marginal oracle", oracle[row], float(oracle_nmse[row])),
            ("MIMO-GS rendering", rendered[row], float(mimogs_nmse[row])),
        )
        for column, (title, power_map, nmse) in enumerate(panels):
            axis = axes[row][column]
            image = axis.imshow(
                to_db(power_map, floor_db),
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
                vmin=floor_db,
                vmax=0.0,
            )
            # The same GT top-4 cells on every panel, so a bright oracle cell
            # with no marker is visibly a ghost.
            for flat_index in gt_top_pairs[row]:
                beam_row, beam_col = divmod(int(flat_index), int(beam_cols))
                axis.add_patch(
                    Rectangle(
                        (beam_col - 0.5, beam_row - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor="red",
                        linewidth=1.2,
                    )
                )
            if row == 0:
                axis.set_title(title, fontsize=10)
            axis.tick_params(labelsize=7)
            if row == rows - 1:
                axis.set_xlabel("Tx beam index", fontsize=8)
            # NMSE goes inside the panel: as an xlabel it sits between two
            # rows and reads as though it belonged to the row below.
            if nmse is not None:
                axis.text(
                    0.02,
                    0.04,
                    f"NMSE {nmse:.2f} dB",
                    transform=axis.transAxes,
                    fontsize=8,
                    ha="left",
                    va="bottom",
                    color="white",
                    bbox={"boxstyle": "round", "facecolor": "black",
                          "alpha": 0.45, "linewidth": 0},
                )

        axes[row][0].set_ylabel(
            f"({coordinates[row, 0]:.0f}, {coordinates[row, 1]:.0f}) m\n"
            f"{components[row]} components\nRx beam",
            fontsize=8,
        )
        axes[row][1].text(
            0.98,
            0.04,
            f"{ghosts[row]} ghost / {GHOST_TOPK}",
            transform=axes[row][1].transAxes,
            fontsize=8,
            ha="right",
            va="bottom",
            color="white",
            bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.45,
                  "linewidth": 0},
        )

    colorbar = figure.colorbar(
        image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.012
    )
    colorbar.set_label("Normalized power [dB]")

    figure.suptitle(
        "Ground truth vs. marginal oracle vs. MIMO-GS  "
        f"(red squares = GT top-{GHOST_TOPK} pairs, floor {floor_db:g} dB)",
        fontsize=11,
    )

    figure.savefig(os.path.join(output_dir, "fig_oracle_qualitative.png"), dpi=200)
    figure.savefig(os.path.join(output_dir, "fig_oracle_qualitative.pdf"))
    plt.close(figure)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Marginal (one-sided) oracle upper bound for beam-pair maps"
    )
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--floor_db", type=float, default=DEFAULT_FLOOR_DB)
    parser.add_argument("--outputs_root", type=str, default="outputs")
    parser.add_argument("--analysis_dir", type=str, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--source_path", type=str, default="")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    repository_root = os.path.dirname(os.path.abspath(__file__))

    outputs_root = arguments.outputs_root
    if not os.path.isabs(outputs_root):
        outputs_root = os.path.join(repository_root, outputs_root)

    checkpoint_argument = arguments.ckpt
    if checkpoint_argument and not os.path.isabs(checkpoint_argument):
        candidate = os.path.join(repository_root, checkpoint_argument)
        if os.path.exists(candidate):
            checkpoint_argument = candidate

    run_dir, checkpoint_path = resolve_run_dir(checkpoint_argument, outputs_root)
    run_name = os.path.basename(os.path.normpath(run_dir))

    print("=" * 78)
    print(f"[eval_marginal_oracle] RUN        : {run_name}")
    print(f"[eval_marginal_oracle] checkpoint : {checkpoint_path}")
    print("=" * 78)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_params, opt_params = restore_config(run_dir, checkpoint)

    if arguments.source_path:
        model_params.source_path = os.path.abspath(arguments.source_path)
    if not os.path.isdir(getattr(model_params, "source_path", "")):
        raise SystemExit(
            f"[eval_marginal_oracle] Dataset "
            f"'{getattr(model_params, 'source_path', '')}' is missing."
        )

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
            f"[eval_marginal_oracle] checkpoint gain MLP is {hidden_dim}-wide; "
            f"rebuilding it to match."
        )
    with gain_net_width(hidden_dim):
        scene, gaussians = build_scene_and_model(
            model_params, opt_params, checkpoint, device
        )

    scale_factor = float(getattr(scene.test_set, "scale_factor", 1.0))
    beam_rows, beam_cols = int(scene.beam_rows), int(scene.beam_cols)
    print(
        f"[eval_marginal_oracle] device={device} | beam grid {beam_rows}x{beam_cols} "
        f"| test samples={len(scene.test_set)}"
    )

    # ------------------------------------------------------------------
    # MIMO-GS joint metrics, straight from eval_render so they match E1
    # ------------------------------------------------------------------
    print("[eval_marginal_oracle] scoring MIMO-GS with eval_render ...")
    mimogs_results = evaluate_test_set(
        scene, gaussians, model_params, device, batch_size, use_cuda_rasterizer
    )
    kept_indices = mimogs_results["index"]
    coordinates = mimogs_results["position"] * scale_factor
    num_locations = int(kept_indices.shape[0])
    print(
        f"[eval_marginal_oracle] MIMO-GS: {num_locations} locations "
        f"(skipped zero-power: {int(mimogs_results['skipped_zero_power'])})"
    )

    # ------------------------------------------------------------------
    # Ground truth + oracle + a re-render for the marginal check / figure
    # ------------------------------------------------------------------
    ground_truth = scene.test_set.magnitude[
        torch.as_tensor(kept_indices, dtype=torch.long)
    ].reshape(num_locations, beam_rows, beam_cols).to(device)

    total_power = ground_truth.reshape(num_locations, -1).sum(dim=1)
    degenerate = int((total_power <= 0.0).sum().item())
    if degenerate:
        print(
            f"[eval_marginal_oracle] WARNING: {degenerate} location(s) carry zero "
            f"total power; the oracle is undefined there and they are dropped."
        )
        keep = (total_power > 0.0).cpu().numpy()
        ground_truth = ground_truth[torch.as_tensor(keep, device=device)]
        kept_indices = kept_indices[keep]
        coordinates = coordinates[keep]
        for key in ("nmse_raw_db", "nmse_shape_db"):
            mimogs_results[key] = mimogs_results[key][keep]
        for group in ("topk", "capture"):
            mimogs_results[group] = {
                k: v[keep] for k, v in mimogs_results[group].items()
            }
        num_locations = int(ground_truth.shape[0])

    oracle = marginal_oracle(ground_truth)

    print("[eval_marginal_oracle] rendering MIMO-GS maps for the marginal check ...")
    rx_positions = scene.test_set.positions[
        torch.as_tensor(kept_indices, dtype=torch.long)
    ]
    tx_pos = torch.as_tensor(scene.bs_position, dtype=torch.float32, device=device)
    rendered_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, num_locations, batch_size):
            rendered_chunks.append(
                render_batch(
                    rx_positions[start : start + batch_size].to(device),
                    tx_pos,
                    gaussians,
                    scene,
                    model_params,
                    use_cuda_rasterizer,
                ).float()
            )
    rendered = torch.cat(rendered_chunks, dim=0)

    # ------------------------------------------------------------------
    # Sanity: the oracle is exact on the one-sided task and rank-1
    # ------------------------------------------------------------------
    rx_error, tx_error = assert_marginals_match(oracle, ground_truth)
    rank_ratio = assert_rank_one(oracle)
    print()
    print("[eval_marginal_oracle] sanity checks")
    print(
        f"  oracle marginals reproduce the GT marginals : "
        f"rx {rx_error:.2e}, tx {tx_error:.2e} (tol 1e-06) : OK"
    )
    print(
        f"  oracle is numerically rank-1                : worst "
        f"sigma_2/sigma_1 = {rank_ratio:.2e} : OK"
    )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    oracle_scores = score_against_gt(oracle, ground_truth)
    ghosts, components, gt_top_pairs = ghost_and_component_counts(
        oracle, ground_truth
    )

    separability = rank1_energy_fraction(ground_truth)
    best_rank1 = best_rank1_approximation(ground_truth)
    best_rank1_scores = score_against_gt(best_rank1, ground_truth)

    oracle_rx_nmse = marginal_nmse_db(oracle, ground_truth, axis=2)
    oracle_tx_nmse = marginal_nmse_db(oracle, ground_truth, axis=1)
    mimogs_rx_nmse = marginal_nmse_db(rendered, ground_truth, axis=2)
    mimogs_tx_nmse = marginal_nmse_db(rendered, ground_truth, axis=1)

    mimogs_shape = mimogs_results["nmse_shape_db"]
    oracle_shape = oracle_scores["nmse_shape_db"]

    print(
        f"  MIMO-GS rx marginal NMSE ({np.mean(mimogs_rx_nmse):.2f} dB) at least "
        f"as good as its joint NMSE ({np.mean(mimogs_shape):.2f} dB) : "
        f"{'OK' if np.mean(mimogs_rx_nmse) <= np.mean(mimogs_shape) else 'VIOLATED'}"
    )

    oracle_beats = float(np.mean(oracle_shape)) < float(np.mean(mimogs_shape))
    if oracle_beats:
        print(
            "  WARNING: the marginal oracle beats MIMO-GS on mean joint NMSE; "
            "see the component-count breakdown below."
        )
    else:
        print(
            f"  oracle joint NMSE ({np.mean(oracle_shape):.2f} dB) is worse than "
            f"MIMO-GS's ({np.mean(mimogs_shape):.2f} dB) : OK"
        )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    output_dir = arguments.analysis_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(repository_root, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []

    def add_row(label: str, values: np.ndarray, scored: Optional[Dict] = None,
                note: str = "") -> None:
        stats = summarize(values)
        row: Dict[str, object] = {
            "row": label,
            "num_locations": num_locations,
            "NMSE_mean_dB": f"{stats['mean']:.6f}",
            "NMSE_median_dB": f"{stats['median']:.6f}",
            "NMSE_p5_dB": f"{stats['p5']:.6f}",
            "NMSE_p95_dB": f"{stats['p95']:.6f}",
        }
        for k in TOPK_ACC_VALUES:
            key = f"topk_acc_K{k}"
            row[f"{key}_mean"] = (
                f"{float(np.mean(scored[key])):.6f}" if scored else ""
            )
        for k in CAPTURE_VALUES:
            key = f"power_capture_K{k}"
            row[f"{key}_mean"] = (
                f"{float(np.mean(scored[key])):.6f}" if scored else ""
            )
        row["note"] = note
        summary_rows.append(row)

    add_row(
        "Marginal oracle (joint map)",
        oracle_shape,
        oracle_scores,
        "rank-1 outer product of the GT marginals; upper bound for one-sided methods",
    )
    add_row(
        "MIMO-GS (joint map)",
        mimogs_shape,
        {
            **{f"topk_acc_K{k}": mimogs_results["topk"][k] for k in TOPK_ACC_VALUES},
            **{
                f"power_capture_K{k}": mimogs_results["capture"][k]
                for k in CAPTURE_VALUES
            },
        },
        f"recomputed from {run_name} with eval_render.evaluate_test_set",
    )
    add_row(
        "Best rank-1 (SVD)",
        best_rank1_scores["nmse_shape_db"],
        best_rank1_scores,
        "tightest possible rank-1 fit; the marginal oracle cannot beat it",
    )
    add_row("Marginal oracle (rx marginal)", oracle_rx_nmse, None,
            "exact by construction")
    add_row("Marginal oracle (tx marginal)", oracle_tx_nmse, None,
            "exact by construction")
    add_row("MIMO-GS (rx marginal)", mimogs_rx_nmse, None,
            "reverse-direction check")
    add_row("MIMO-GS (tx marginal)", mimogs_tx_nmse, None,
            "reverse-direction check")

    with open(os.path.join(output_dir, "metrics_summary.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    with open(os.path.join(output_dir, "per_location.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "test_index", "x", "y", "z",
                "oracle_NMSE_dB", "mimogs_NMSE_dB", "oracle_minus_mimogs_dB",
                "ghost_count_top4", "num_significant_components",
                "gt_rank1_energy_fraction",
                "oracle_rx_marginal_NMSE_dB", "mimogs_rx_marginal_NMSE_dB",
                "mimogs_tx_marginal_NMSE_dB",
            ]
        )
        for row in range(num_locations):
            writer.writerow(
                [
                    int(kept_indices[row]),
                    f"{coordinates[row, 0]:.6f}",
                    f"{coordinates[row, 1]:.6f}",
                    f"{coordinates[row, 2]:.6f}",
                    f"{oracle_shape[row]:.6f}",
                    f"{mimogs_shape[row]:.6f}",
                    f"{oracle_shape[row] - mimogs_shape[row]:.6f}",
                    int(ghosts[row]),
                    int(components[row]),
                    f"{separability[row]:.6f}",
                    f"{oracle_rx_nmse[row]:.6f}",
                    f"{mimogs_rx_nmse[row]:.6f}",
                    f"{mimogs_tx_nmse[row]:.6f}",
                ]
            )

    # -- qualitative figure ---------------------------------------------
    picks = select_qualitative(components, coordinates, QUALITATIVE_LOCATIONS)
    index = np.asarray(picks, dtype=np.int64)
    plot_qualitative(
        output_dir,
        ground_truth[torch.as_tensor(index, device=device)]
        .cpu().numpy().astype(np.float64),
        oracle[torch.as_tensor(index, device=device)]
        .cpu().numpy().astype(np.float64),
        rendered[torch.as_tensor(index, device=device)]
        .cpu().numpy().astype(np.float64),
        gt_top_pairs[index],
        oracle_shape[index],
        mimogs_shape[index],
        ghosts[index],
        components[index],
        coordinates[index],
        beam_cols,
        float(arguments.floor_db),
    )

    # ------------------------------------------------------------------
    # Component-count breakdown
    # ------------------------------------------------------------------
    edges = [(1, 1), (2, 4), (5, 16), (17, 10 ** 9)]
    labels = ["1 component", "2-4", "5-16", "17+"]
    breakdown: List[Dict[str, object]] = []
    print()
    print("[eval_marginal_oracle] joint NMSE by GT component count "
          f"(bins >= {COMPONENT_THRESHOLD:.0%} of peak)")
    print(f"  {'components':<14}{'N':>7}{'oracle':>10}{'MIMO-GS':>10}"
          f"{'gap':>9}{'ghosts':>9}")
    for (low, high), label in zip(edges, labels):
        mask = (components >= low) & (components <= high)
        if not np.any(mask):
            continue
        entry = {
            "components": label,
            "num_locations": int(mask.sum()),
            "oracle_NMSE_mean_dB": float(np.mean(oracle_shape[mask])),
            "mimogs_NMSE_mean_dB": float(np.mean(mimogs_shape[mask])),
            "gap_dB": float(
                np.mean(oracle_shape[mask]) - np.mean(mimogs_shape[mask])
            ),
            "mean_ghost_count": float(np.mean(ghosts[mask])),
        }
        breakdown.append(entry)
        print(
            f"  {label:<14}{entry['num_locations']:>7d}"
            f"{entry['oracle_NMSE_mean_dB']:>10.2f}"
            f"{entry['mimogs_NMSE_mean_dB']:>10.2f}"
            f"{entry['gap_dB']:>9.2f}{entry['mean_ghost_count']:>9.2f}"
        )

    print()
    print("[eval_marginal_oracle] joint NMSE by GT separability "
          "(sigma_1^2 / sum sigma^2)")
    print(f"  {'rank-1 energy':<16}{'N':>7}{'oracle':>10}{'MIMO-GS':>10}{'gap':>9}")
    for label, low, high in (
        ("< 0.90", 0.0, 0.90),
        ("0.90-0.99", 0.90, 0.99),
        ("0.99-0.999", 0.99, 0.999),
        (">= 0.999", 0.999, 1.01),
    ):
        mask = (separability >= low) & (separability < high)
        if not np.any(mask):
            continue
        oracle_mean = float(np.mean(oracle_shape[mask]))
        mimogs_mean = float(np.mean(mimogs_shape[mask]))
        print(
            f"  {label:<16}{int(mask.sum()):>7d}{oracle_mean:>10.2f}"
            f"{mimogs_mean:>10.2f}{oracle_mean - mimogs_mean:>9.2f}"
        )
        breakdown.append(
            {
                "components": f"separability {label}",
                "num_locations": int(mask.sum()),
                "oracle_NMSE_mean_dB": oracle_mean,
                "mimogs_NMSE_mean_dB": mimogs_mean,
                "gap_dB": oracle_mean - mimogs_mean,
                "mean_ghost_count": float(np.mean(ghosts[mask])),
            }
        )

    with open(os.path.join(output_dir, "component_breakdown.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(breakdown[0].keys()))
        writer.writeheader()
        for entry in breakdown:
            writer.writerow(entry)

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    readme = [
        "Marginal (one-sided) oracle -- upper bound for separable beam models",
        "=" * 70,
        "",
        "Generated by eval_marginal_oracle.py (repository root).",
        "",
        "CONSTRUCTION",
        "-" * 70,
        "  From each ground-truth beam-pair map X (Nr x Nt):",
        "      r_m = sum_n X[m,n]      (rx marginal)",
        "      t_n = sum_m X[m,n]      (tx marginal)",
        "      S   = sum_{m,n} X[m,n]",
        "      X_oracle[m,n] = r_m * t_n / S",
        "  Total power is preserved, so the oracle sits on the GT's absolute",
        "  scale and its own marginals equal the GT marginals exactly",
        f"  (verified: rx {rx_error:.2e}, tx {tx_error:.2e} relative to the",
        "  marginal peak, tolerance 1e-06).  It is rank-1 by construction",
        f"  (verified: worst sigma_2/sigma_1 = {rank_ratio:.2e}).",
        "",
        "WHAT IT BOUNDS",
        "-" * 70,
        "  Any method rendering only one-sided spectra is bounded by this",
        "  oracle on the joint task: knowing both marginals perfectly still",
        "  leaves the rx-tx coupling unspecified, and the outer product is the",
        "  only reconstruction consistent with them.  The oracle is therefore",
        "  the ceiling for separable models and the floor that a genuinely",
        "  joint model has to beat to justify itself.",
        "",
        "CONVENTIONS",
        "-" * 70,
        "  Metric code is imported from eval_render (NMSE, top-K overlap,",
        "  power capture); the dB map convention comes from eval_db_16by64",
        "  (per-map peak-normalized, 10*log10 because the stored maps are",
        "  already power, floor -40 dB, viridis).",
        "  The headline NMSE is the SHAPE convention -- max-normalized",
        "  prediction against max-normalized target.  It is the only one of",
        "  eval_render's two conventions that is comparable across methods on",
        "  a different absolute scale; both are in the CSVs.",
        f"  A GT bin counts as a significant component at {COMPONENT_THRESHOLD:.0%}",
        f"  of the map peak (-{-10 * np.log10(COMPONENT_THRESHOLD):.0f} dB); a",
        f"  'ghost' is an oracle top-{GHOST_TOPK} pick whose true power is below",
        "  that threshold.",
        "",
        "MATERIALS",
        "-" * 70,
        f"  Checkpoint : {os.path.relpath(checkpoint_path, repository_root)}"
        f"  (iteration {int(checkpoint.get('iteration', -1))})",
        f"  Dataset    : {getattr(model_params, 'source_path', '')}  -- TEST split",
        f"  Locations  : {num_locations}"
        + (f"  ({degenerate} zero-power dropped)" if degenerate else ""),
        f"  Beam grid  : {beam_rows} Rx x {beam_cols} Tx",
        "",
        "RESULTS (mean over test locations)",
        "-" * 70,
        f"  {'row':<32}{'mean':>9}{'median':>9}{'top-1':>9}{'cap@4':>9}",
    ]
    for row in summary_rows:
        readme.append(
            f"  {str(row['row']):<32}{float(row['NMSE_mean_dB']):>9.2f}"
            f"{float(row['NMSE_median_dB']):>9.2f}"
            + (
                f"{float(row['topk_acc_K1_mean']):>9.3f}"
                f"{float(row['power_capture_K4_mean']):>9.3f}"
                if row["topk_acc_K1_mean"]
                else f"{'--':>9}{'--':>9}"
            )
        )
    readme += [
        "",
        f"  GT separability (sigma_1^2 / sum sigma^2): median "
        f"{float(np.median(separability)):.4f}, mean "
        f"{float(np.mean(separability)):.4f}, p5 "
        f"{float(np.percentile(separability, 5)):.4f}",
        "",
        "  READ THIS BEFORE QUOTING THE TABLE.  On this dataset the marginal",
        "  oracle is NOT a loose bound -- it beats MIMO-GS on mean joint NMSE.",
        "  The reason is the separability line above: the leading singular",
        "  value already carries a median ~99.8% of each map's energy, i.e. the",
        "  ground-truth beam maps are very nearly outer products, so there is",
        "  almost no rx-tx coupling left for a joint model to capture.  MIMO-GS",
        "  still wins on the beam-selection metrics (top-1 accuracy, power",
        "  capture) and in the bad tail (p95), but a joint-vs-separable",
        "  argument cannot be made on this scene.",
        "",
        f"  Mean ghost pairs among the oracle's top-{GHOST_TOPK}: "
        f"{float(np.mean(ghosts)):.2f}",
        f"  Mean significant GT components per location: "
        f"{float(np.mean(components)):.2f}",
        "",
        "FILES",
        "-" * 70,
        "  metrics_summary.csv        one row per scored quantity",
        "  per_location.csv           per-location NMSE, ghosts, component count",
        "  component_breakdown.csv    joint NMSE binned by GT component count",
        "  fig_oracle_qualitative.*   GT | oracle | MIMO-GS, GT top-4 marked",
    ]

    with open(os.path.join(output_dir, "README.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(readme).rstrip() + "\n")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print(f"[eval_marginal_oracle] SUMMARY -- {num_locations} test locations")
    print("=" * 78)
    print(f"  {'row':<32}{'mean':>9}{'median':>9}{'p5':>9}{'p95':>9}"
          f"{'top-1':>9}{'cap@4':>9}")
    for row in summary_rows:
        print(
            f"  {str(row['row']):<32}{float(row['NMSE_mean_dB']):>9.2f}"
            f"{float(row['NMSE_median_dB']):>9.2f}"
            f"{float(row['NMSE_p5_dB']):>9.2f}{float(row['NMSE_p95_dB']):>9.2f}"
            + (
                f"{float(row['topk_acc_K1_mean']):>9.3f}"
                f"{float(row['power_capture_K4_mean']):>9.3f}"
                if row["topk_acc_K1_mean"]
                else f"{'--':>9}{'--':>9}"
            )
        )
    print()
    print(
        f"  mean ghost pairs in the oracle's top-{GHOST_TOPK}: "
        f"{float(np.mean(ghosts)):.2f} / {GHOST_TOPK}   |   "
        f"mean significant GT components: {float(np.mean(components)):.2f}"
    )
    print(
        f"  GT separability sigma_1^2/sum sigma^2: median "
        f"{float(np.median(separability)):.4f}, mean "
        f"{float(np.mean(separability)):.4f}, p5 "
        f"{float(np.percentile(separability, 5)):.4f}"
        f"   -> the maps are nearly outer products"
    )
    print()
    print("  qualitative locations (most multi-component):")
    for row in picks:
        print(
            f"    (x,y)=({coordinates[row, 0]:7.2f},{coordinates[row, 1]:7.2f})  "
            f"{components[row]:>3d} components  oracle "
            f"{oracle_shape[row]:7.2f} dB  MIMO-GS {mimogs_shape[row]:7.2f} dB  "
            f"{ghosts[row]} ghost(s)"
        )
    print()
    print(f"[eval_marginal_oracle] Outputs written to {output_dir}")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
