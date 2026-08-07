"""E4 -- beam management evaluation for MIMO-GS.

Zero-argument runnable::

    python eval_beam_management.py

Approximation (important)
-------------------------
The dataset ``.mat`` files store per-location beam-pair POWER maps, not complex
channel matrices, so the paper's exact log-det net rate over ``H_sel`` cannot be
formed. This script evaluates the PARALLEL-SUBCHANNEL APPROXIMATION instead --
the same functional form as the paper's selection metric ``f(B; p)``::

    R(p; L_t) = sum_{m in B_r} log2(1 + (P / (L_t * sigma^2)) * sum_{n in B_t} X[m, n])

Beam SELECTION is driven by the RENDERED map, while the RATE is always
evaluated on the GROUND-TRUTH map. That mirrors the paper's protocol, where
CSI-RS measurement only happens on the subspace the prior already selected.

Scale convention
----------------
Raw per-location peak power spans roughly three orders of magnitude across the
test set, and the renderer is trained against the per-location max-normalized
target (``utils/loss.py::normalize_mag_map``, the scale term of
``composite_magnitude_loss``). Both the rendered and the ground-truth map are
therefore max-normalized per location before any selection or rate evaluation,
so ``--snr_db`` is a PEAK SNR relative to the strongest beam pair at that
location, identical for prediction and ground truth.
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

from utils.loss import normalize_mag_map

try:
    import eval_render
except ImportError as error:  # pragma: no cover - eval_render ships with E1.
    raise SystemExit(
        "[eval_bm] eval_render.py is required for the checkpoint-loading and "
        f"rendering helpers but could not be imported: {error}"
    )


EPS = 1e-12
RANDOM_DRAWS = 20
RANDOM_SEED = 12345
CDF_BUDGET = 8
SELECTION_CHUNK = 512


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def render_test_set(
    scene,
    gaussians,
    model_params,
    device: torch.device,
    batch_size: int,
    use_cuda_rasterizer: bool,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, int]:
    """Render every test location and return normalized prediction/GT maps."""
    from torch.utils.data import DataLoader

    total = len(scene.test_set)
    if total == 0:
        raise SystemExit("[eval_bm] The test set is empty; nothing to evaluate.")

    loader = DataLoader(
        scene.test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    tx_pos = torch.as_tensor(scene.bs_position, dtype=torch.float32, device=device)

    predicted_maps: List[torch.Tensor] = []
    truth_maps: List[torch.Tensor] = []
    positions: List[np.ndarray] = []
    indices: List[int] = []
    skipped = 0
    cursor = 0

    with torch.no_grad():
        for magnitude, rx_pos in loader:
            magnitude = magnitude.to(device, non_blocking=True)
            rx_pos = rx_pos.to(device, non_blocking=True)

            batch = magnitude.shape[0]
            batch_indices = torch.arange(cursor, cursor + batch)
            cursor += batch

            ground_truth = magnitude.reshape(batch, scene.beam_rows, scene.beam_cols)

            peak = ground_truth.reshape(batch, -1).amax(dim=1)
            valid = peak > EPS
            skipped += batch - int(valid.sum().item())
            if not bool(valid.any()):
                continue

            predicted = eval_render.render_batch(
                rx_pos, tx_pos, gaussians, scene, model_params, use_cuda_rasterizer
            )

            ground_truth = ground_truth[valid]
            predicted = predicted[valid].clamp_min(0.0)

            truth_maps.append(normalize_mag_map(ground_truth))
            predicted_maps.append(normalize_mag_map(predicted))
            positions.append(rx_pos.reshape(batch, 3)[valid].cpu().numpy())
            indices.extend(batch_indices[valid.cpu()].tolist())

    if not indices:
        raise SystemExit("[eval_bm] Every test map had zero power.")

    return (
        torch.cat(predicted_maps, dim=0),
        torch.cat(truth_maps, dim=0),
        np.concatenate(positions, axis=0),
        np.asarray(indices, dtype=np.int64),
        skipped,
    )


# ----------------------------------------------------------------------
# Selection rule (paper Section VI)
# ----------------------------------------------------------------------
def selection_metric(row_sums: torch.Tensor, coefficient: float, num_rx: int):
    """``max_{|R|=L_r} sum_{m in R} log2(1 + c * s_m)`` over the last axis.

    The inner maximization is just "keep the L_r largest rows", because
    ``log2(1 + c*s)`` is monotone in ``s``.
    """
    utility = torch.log2(1.0 + coefficient * row_sums)
    return torch.topk(utility, k=num_rx, dim=-1).values.sum(dim=-1)


def greedy_select(
    beam_map: torch.Tensor, budget: int, num_rx: int, snr_linear: float
) -> torch.Tensor:
    """Greedy transmit-beam selection maximizing ``f(B; p)``.

    Args:
        beam_map: ``(B, Nr, Nt)`` non-negative power maps.
        budget: ``L_t``.
    Returns:
        ``(B, L_t)`` selected transmit beam indices.

    Each step adds the beam with the largest metric increase; since ``f(B)`` is
    identical across candidates at a given step, that is the same as taking the
    argmax of ``f(B + {n})``. The power split ``P / (L_t sigma^2)`` uses the
    target budget ``L_t``, which is what ``f(B; p)`` is parameterized by.
    """
    num_locations, num_rx_beams, num_tx_beams = beam_map.shape
    budget = int(min(budget, num_tx_beams))

    if budget >= num_tx_beams:
        # Only one subset of size Nt exists, so every scheme selects it.
        return (
            torch.arange(num_tx_beams, device=beam_map.device)
            .unsqueeze(0)
            .expand(num_locations, -1)
            .contiguous()
        )

    coefficient = float(snr_linear) / float(budget)
    row_sums = beam_map.new_zeros(num_locations, num_rx_beams)
    taken = torch.zeros(
        num_locations, num_tx_beams, dtype=torch.bool, device=beam_map.device
    )
    chosen = torch.zeros(
        num_locations, budget, dtype=torch.long, device=beam_map.device
    )

    transposed = beam_map.permute(0, 2, 1).contiguous()  # (B, Nt, Nr)

    for step in range(budget):
        # candidate[b, n, m] = current row sum + contribution of beam n
        candidate = row_sums.unsqueeze(1) + transposed
        scores = selection_metric(candidate, coefficient, num_rx)
        scores = scores.masked_fill(taken, float("-inf"))

        best = scores.argmax(dim=1)
        chosen[:, step] = best
        taken.scatter_(1, best.unsqueeze(1), True)
        row_sums = row_sums + transposed.gather(
            1, best.view(-1, 1, 1).expand(-1, 1, num_rx_beams)
        ).squeeze(1)

    return chosen


def evaluate_rate(
    truth_map: torch.Tensor,
    tx_beams: torch.Tensor,
    budget: int,
    num_rx: int,
    snr_linear: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Receive selection and rate, both on the GROUND-TRUTH map.

    Returns ``(rate, rx_beams)``. The UE measures ``s_m`` only on the
    transmit beams that were configured, then keeps the ``L_r`` strongest.
    """
    num_locations, num_rx_beams, _ = truth_map.shape
    coefficient = float(snr_linear) / float(budget)

    gather_index = tx_beams.unsqueeze(1).expand(-1, num_rx_beams, -1)
    row_sums = truth_map.gather(2, gather_index).sum(dim=2)  # (B, Nr)

    rx_beams = torch.topk(row_sums, k=num_rx, dim=1).indices
    selected_sums = row_sums.gather(1, rx_beams)
    rate = torch.log2(1.0 + coefficient * selected_sums).sum(dim=1)
    return rate, rx_beams


def random_select(
    num_locations: int,
    num_tx_beams: int,
    budget: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """One uniformly random transmit subset of size ``budget`` per location."""
    scores = torch.rand(num_locations, num_tx_beams, generator=generator)
    return torch.topk(scores, k=int(budget), dim=1).indices.to(device)


# ----------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------
def write_csv(path: str, header: Sequence[str], rows: Sequence[Sequence]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def save_figure(figure, output_dir: str, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, f"{stem}.png"), dpi=200)
    figure.savefig(os.path.join(output_dir, f"{stem}.pdf"))
    plt.close(figure)


def plot_rate_curves(
    output_dir: str,
    stem: str,
    title: str,
    ylabel: str,
    budgets: Sequence[int],
    curves: Dict[str, Sequence[float]],
    reference_value: float,
    reference_label: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.8))

    styles = {
        "MIMO-GS": ("o", "tab:blue", "-"),
        "Genie": ("s", "tab:green", "-"),
        "Random": ("^", "tab:red", "--"),
    }
    for name, values in curves.items():
        marker, color, linestyle = styles.get(name, ("o", None, "-"))
        axis.plot(budgets, values, marker=marker, color=color, linestyle=linestyle,
                  linewidth=1.7, label=name)

    axis.axhline(reference_value, color="0.35", linestyle=":", linewidth=1.6,
                 label=reference_label)

    axis.set_xscale("log", base=2)
    axis.set_xticks(list(budgets))
    axis.set_xticklabels([str(b) for b in budgets])
    axis.set_xlabel(r"transmit budget $L_t$")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.3, linewidth=0.5)
    axis.legend(fontsize=8)

    save_figure(figure, output_dir, stem)


def plot_alignment(
    output_dir: str, budgets: Sequence[int], curves: Dict[str, Sequence[float]]
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.6))

    axis.plot(budgets, curves["MIMO-GS"], marker="o", color="tab:blue",
              linewidth=1.7, label="MIMO-GS")
    axis.plot(budgets, curves["Random"], marker="^", color="tab:red",
              linestyle="--", linewidth=1.7, label="Random")
    axis.axhline(1.0, color="0.35", linestyle=":", linewidth=1.4, label="Genie")

    axis.set_xscale("log", base=2)
    axis.set_xticks(list(budgets))
    axis.set_xticklabels([str(b) for b in budgets])
    axis.set_xlabel(r"transmit budget $L_t$")
    axis.set_ylabel("alignment efficiency  rate / rate(genie)")
    axis.set_title("Alignment efficiency vs. transmit budget")
    axis.grid(alpha=0.3, linewidth=0.5)
    axis.set_ylim(0.0, 1.05)
    axis.legend(fontsize=8)

    save_figure(figure, output_dir, "fig_alignment_vs_Lt")


def plot_rate_cdf(output_dir: str, budget: int, rates: Dict[str, np.ndarray]) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 4.6))

    styles = {
        "MIMO-GS": "tab:blue",
        "Genie": "tab:green",
        "Random": "tab:red",
    }
    for name, values in rates.items():
        ordered = np.sort(values)
        probabilities = np.arange(1, ordered.shape[0] + 1) / ordered.shape[0]
        axis.plot(ordered, probabilities, linewidth=1.7,
                  color=styles.get(name), label=name)

    axis.set_xlabel("per-location rate [bit/s/Hz]")
    axis.set_ylabel("empirical CDF")
    axis.set_title(f"Per-location rate distribution at $L_t$ = {budget}")
    axis.grid(alpha=0.3, linewidth=0.5)
    axis.set_ylim(0.0, 1.0)
    axis.legend(fontsize=8)

    save_figure(figure, output_dir, "fig_rate_cdf")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MIMO-GS beam management evaluation (E4)"
    )
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--outputs_root", type=str, default="outputs")
    parser.add_argument("--analysis_root", type=str, default="analysis")
    parser.add_argument("--Lr", type=int, default=2, help="receive budget L_r")
    parser.add_argument("--snr_db", type=float, default=10.0,
                        help="peak P/sigma^2 in dB")
    parser.add_argument("--Lt_grid", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--batch_size", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    repository_root = os.path.dirname(os.path.abspath(__file__))

    outputs_root = arguments.outputs_root
    if not os.path.isabs(outputs_root):
        outputs_root = os.path.join(repository_root, outputs_root)

    run_dir, checkpoint_path = eval_render.resolve_run_dir(arguments.ckpt, outputs_root)
    run_name = os.path.basename(os.path.normpath(run_dir))

    print("=" * 78)
    print(f"[eval_bm] EVALUATING RUN : {run_name}")
    print(f"[eval_bm] run directory  : {run_dir}")
    print(f"[eval_bm] checkpoint     : {checkpoint_path}")
    print("=" * 78)
    print("[eval_bm] APPROXIMATION: the dataset stores beam-pair POWER maps, not")
    print("[eval_bm]   complex channels, so the exact log-det net rate over H_sel")
    print("[eval_bm]   cannot be formed. Rates below use the parallel-subchannel")
    print("[eval_bm]   approximation R = sum_m log2(1 + (P/(L_t sigma^2)) sum_n X[m,n]),")
    print("[eval_bm]   the same functional form as the paper's selection metric f(B;p).")
    print("[eval_bm]   SELECTION uses the rendered map; RATE is always evaluated on")
    print("[eval_bm]   ground truth. Both maps are per-location max-normalized, so")
    print("[eval_bm]   --snr_db is a PEAK SNR.")
    print("=" * 78)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_params, opt_params = eval_render.restore_config(run_dir, checkpoint)

    if not os.path.isdir(getattr(model_params, "source_path", "")):
        raise SystemExit(
            f"[eval_bm] Dataset directory "
            f"'{getattr(model_params, 'source_path', '')}' is missing."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda_rasterizer = bool(
        int(getattr(model_params, "use_cuda_rasterizer", 1))
    ) and device.type == "cuda"
    batch_size = max(
        1, int(arguments.batch_size) or int(getattr(model_params, "batch_size", 8))
    )

    scene, gaussians = eval_render.build_scene_and_model(
        model_params, opt_params, checkpoint, device
    )

    predicted, truth, positions, indices, skipped = render_test_set(
        scene, gaussians, model_params, device, batch_size, use_cuda_rasterizer
    )

    num_locations, num_rx_beams, num_tx_beams = truth.shape
    num_rx = max(1, min(int(arguments.Lr), num_rx_beams))
    snr_linear = float(10.0 ** (arguments.snr_db / 10.0))

    budgets = sorted({max(1, min(int(b), num_tx_beams)) for b in arguments.Lt_grid})
    # Guarantee the full-sweep anchor is present so the reference line is exact.
    if num_tx_beams not in budgets:
        budgets.append(num_tx_beams)
        budgets = sorted(budgets)

    # tau/T_B is fixed so that a full sweep (L_t = Nt) costs half the interval.
    tau_over_tb = 0.5 / float(num_tx_beams)

    print(f"[eval_bm] locations={num_locations} (skipped zero-power: {skipped}) | "
          f"Nr={num_rx_beams} Nt={num_tx_beams}")
    print(f"[eval_bm] L_r={num_rx} | SNR={arguments.snr_db:.1f} dB "
          f"(linear {snr_linear:.3f}) | tau/T_B={tau_over_tb:.6f}")
    print(f"[eval_bm] L_t grid: {budgets}")
    print("")

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    rate_tables: Dict[str, Dict[int, np.ndarray]] = {
        "MIMO-GS": {}, "Genie": {}, "Random": {}
    }
    selection_at_cdf: Dict[str, np.ndarray] = {}
    beams_at_cdf: Dict[str, np.ndarray] = {}

    for budget in budgets:
        gs_rates = torch.empty(num_locations, device=device)
        genie_rates = torch.empty(num_locations, device=device)
        random_rates = torch.zeros(num_locations, device=device)

        gs_tx = torch.empty(num_locations, budget, dtype=torch.long, device=device)
        genie_tx = torch.empty(num_locations, budget, dtype=torch.long, device=device)
        gs_rx = torch.empty(num_locations, num_rx, dtype=torch.long, device=device)
        genie_rx = torch.empty(num_locations, num_rx, dtype=torch.long, device=device)

        for start in range(0, num_locations, SELECTION_CHUNK):
            stop = min(start + SELECTION_CHUNK, num_locations)
            predicted_chunk = predicted[start:stop]
            truth_chunk = truth[start:stop]

            tx_gs = greedy_select(predicted_chunk, budget, num_rx, snr_linear)
            tx_genie = greedy_select(truth_chunk, budget, num_rx, snr_linear)

            rate_gs, rx_gs = evaluate_rate(
                truth_chunk, tx_gs, budget, num_rx, snr_linear
            )
            rate_genie, rx_genie = evaluate_rate(
                truth_chunk, tx_genie, budget, num_rx, snr_linear
            )

            gs_rates[start:stop] = rate_gs
            genie_rates[start:stop] = rate_genie
            gs_tx[start:stop] = tx_gs
            genie_tx[start:stop] = tx_genie
            gs_rx[start:stop] = rx_gs
            genie_rx[start:stop] = rx_genie

        for _ in range(RANDOM_DRAWS):
            draw_rates = torch.empty(num_locations, device=device)
            for start in range(0, num_locations, SELECTION_CHUNK):
                stop = min(start + SELECTION_CHUNK, num_locations)
                tx_random = random_select(
                    stop - start, num_tx_beams, budget, generator, device
                )
                draw_rates[start:stop], _ = evaluate_rate(
                    truth[start:stop], tx_random, budget, num_rx, snr_linear
                )
            random_rates += draw_rates
        random_rates /= float(RANDOM_DRAWS)

        rate_tables["MIMO-GS"][budget] = gs_rates.cpu().numpy()
        rate_tables["Genie"][budget] = genie_rates.cpu().numpy()
        rate_tables["Random"][budget] = random_rates.cpu().numpy()

        if budget == min(CDF_BUDGET, num_tx_beams):
            selection_at_cdf = {
                "MIMO-GS": rate_tables["MIMO-GS"][budget],
                "Genie": rate_tables["Genie"][budget],
                "Random": rate_tables["Random"][budget],
            }
            beams_at_cdf = {
                "tx_mimogs": gs_tx.cpu().numpy(),
                "tx_genie": genie_tx.cpu().numpy(),
                "rx_mimogs": gs_rx.cpu().numpy(),
                "rx_genie": genie_rx.cpu().numpy(),
            }

        print(
            f"  L_t={budget:>3} | MIMO-GS {gs_rates.mean().item():7.4f} | "
            f"Genie {genie_rates.mean().item():7.4f} | "
            f"Random {random_rates.mean().item():7.4f}  [bit/s/Hz]"
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    output_dir = os.path.join(repository_root, arguments.analysis_root, run_name,
                              "beam_management")
    os.makedirs(output_dir, exist_ok=True)

    mean_rates = {
        scheme: [float(np.mean(rate_tables[scheme][b])) for b in budgets]
        for scheme in rate_tables
    }
    prelog = [max(0.0, 1.0 - b * tau_over_tb) for b in budgets]
    net_rates = {
        scheme: [value * factor for value, factor in zip(mean_rates[scheme], prelog)]
        for scheme in mean_rates
    }

    full_sweep_rate = float(np.mean(rate_tables["Genie"][num_tx_beams]))
    full_sweep_net = full_sweep_rate * max(0.0, 1.0 - num_tx_beams * tau_over_tb)

    write_csv(
        os.path.join(output_dir, "rate_vs_Lt.csv"),
        ["L_t", "prelog", "rate_mimogs", "rate_genie", "rate_random",
         "netrate_mimogs", "netrate_genie", "netrate_random"],
        [
            [
                budgets[i], f"{prelog[i]:.6f}",
                f"{mean_rates['MIMO-GS'][i]:.6f}",
                f"{mean_rates['Genie'][i]:.6f}",
                f"{mean_rates['Random'][i]:.6f}",
                f"{net_rates['MIMO-GS'][i]:.6f}",
                f"{net_rates['Genie'][i]:.6f}",
                f"{net_rates['Random'][i]:.6f}",
            ]
            for i in range(len(budgets))
        ],
    )

    # Alignment efficiency: per-location ratio to genie, then averaged.
    alignment: Dict[str, List[float]] = {"MIMO-GS": [], "Random": []}
    degenerate_locations = 0
    for budget in budgets:
        genie = rate_tables["Genie"][budget]
        usable = genie > EPS
        degenerate_locations = max(degenerate_locations, int((~usable).sum()))
        for scheme in alignment:
            ratio = rate_tables[scheme][budget][usable] / genie[usable]
            alignment[scheme].append(float(np.mean(ratio)))

    write_csv(
        os.path.join(output_dir, "alignment_vs_Lt.csv"),
        ["L_t", "alignment_mimogs", "alignment_random"],
        [
            [budgets[i], f"{alignment['MIMO-GS'][i]:.6f}",
             f"{alignment['Random'][i]:.6f}"]
            for i in range(len(budgets))
        ],
    )

    cdf_budget = min(CDF_BUDGET, num_tx_beams)
    scale_factor = float(getattr(scene.test_set, "scale_factor", 1.0))
    coordinates = positions * scale_factor

    def join_beams(row: np.ndarray) -> str:
        return ";".join(str(int(value)) for value in np.sort(row))

    write_csv(
        os.path.join(output_dir, "per_location_L8.csv"),
        ["index", "x", "y", "z", "rate_mimogs", "rate_genie", "rate_random",
         "alignment_mimogs", "tx_beams_mimogs", "tx_beams_genie",
         "rx_beams_mimogs", "rx_beams_genie"],
        [
            [
                int(indices[i]),
                f"{coordinates[i, 0]:.4f}",
                f"{coordinates[i, 1]:.4f}",
                f"{coordinates[i, 2]:.4f}",
                f"{selection_at_cdf['MIMO-GS'][i]:.6f}",
                f"{selection_at_cdf['Genie'][i]:.6f}",
                f"{selection_at_cdf['Random'][i]:.6f}",
                f"{selection_at_cdf['MIMO-GS'][i] / max(selection_at_cdf['Genie'][i], EPS):.6f}",
                join_beams(beams_at_cdf["tx_mimogs"][i]),
                join_beams(beams_at_cdf["tx_genie"][i]),
                join_beams(beams_at_cdf["rx_mimogs"][i]),
                join_beams(beams_at_cdf["rx_genie"][i]),
            ]
            for i in range(num_locations)
        ],
    )

    plot_rate_curves(
        output_dir, "fig_rate_vs_Lt",
        f"Mean rate vs. transmit budget  ($L_r$={num_rx}, "
        f"{arguments.snr_db:.0f} dB peak SNR)",
        "mean rate [bit/s/Hz]", budgets, mean_rates,
        full_sweep_rate, f"full sweep ($L_t$={num_tx_beams}, genie)",
    )
    plot_rate_curves(
        output_dir, "fig_netrate_vs_Lt",
        rf"Mean NET rate vs. transmit budget  ($\tau/T_B$={tau_over_tb:.5f})",
        "mean net rate [bit/s/Hz]", budgets, net_rates,
        full_sweep_net, f"full sweep ($L_t$={num_tx_beams}, genie)",
    )
    plot_alignment(output_dir, budgets, alignment)
    plot_rate_cdf(output_dir, cdf_budget, selection_at_cdf)

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    tolerance = 1e-6
    violations: List[str] = []

    for i, budget in enumerate(budgets):
        if mean_rates["Genie"][i] + tolerance < mean_rates["MIMO-GS"][i]:
            violations.append(
                f"(i) Genie < MIMO-GS at L_t={budget}: "
                f"{mean_rates['Genie'][i]:.6f} < {mean_rates['MIMO-GS'][i]:.6f}"
            )
        if mean_rates["MIMO-GS"][i] + tolerance < mean_rates["Random"][i]:
            violations.append(
                f"(i) MIMO-GS < Random at L_t={budget}: "
                f"{mean_rates['MIMO-GS'][i]:.6f} < {mean_rates['Random'][i]:.6f}"
            )

    last = len(budgets) - 1
    if budgets[last] == num_tx_beams:
        spread = max(mean_rates[s][last] for s in mean_rates) - min(
            mean_rates[s][last] for s in mean_rates
        )
        if spread > 1e-5:
            violations.append(
                f"(ii) schemes disagree at L_t={num_tx_beams} by {spread:.3e}"
            )

    monotonic_breaks = [
        f"L_t {budgets[i]}->{budgets[i + 1]}: "
        f"{mean_rates['Genie'][i]:.4f} -> {mean_rates['Genie'][i + 1]:.4f}"
        for i in range(len(budgets) - 1)
        if mean_rates["Genie"][i + 1] + tolerance < mean_rates["Genie"][i]
    ]

    print("")
    print("=" * 78)
    print("[eval_bm] SANITY CHECKS")
    print("=" * 78)
    print(f"  (i)   Genie >= MIMO-GS >= Random at every L_t : "
          f"{'PASS' if not violations else 'SEE BELOW'}")
    print(f"  (ii)  all schemes agree at L_t = Nt           : "
          f"{'PASS' if all('(ii)' not in v for v in violations) else 'FAIL'}")
    print(f"  (iii) genie rate non-decreasing in L_t        : "
          f"{'PASS' if not monotonic_breaks else 'FAIL'}")
    for violation in violations:
        print(f"    VIOLATION {violation}")
    for entry in monotonic_breaks:
        print(f"    (iii) decrease at {entry}")
    if monotonic_breaks:
        print("    NOTE: the rate metric divides the transmit power by L_t, so a")
        print("    larger budget spreads P over more beams. Once the budget exceeds")
        print("    the number of genuinely distinct paths the L_r receive beams can")
        print("    use, the 1/L_t power split outweighs the extra collected energy")
        print("    and the rate falls. This is a property of the specified metric,")
        print("    not a selection bug -- genie is still the per-L_t upper bound.")

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    readme_lines = [
        "MIMO-GS beam management evaluation (E4)",
        "=" * 60,
        "",
        "APPROXIMATION",
        "-------------",
        "The dataset .mat files contain per-location beam-pair POWER maps, not",
        "complex channel matrices. The paper's exact log-det net rate over H_sel",
        "therefore cannot be computed. This evaluation uses the",
        "parallel-subchannel approximation, which is the same functional form as",
        "the paper's selection metric f(B; p):",
        "",
        "    R(p; L_t) = sum_{m in B_r} log2(1 + (P/(L_t sigma^2)) * "
        "sum_{n in B_t} X[m, n])",
        "",
        "Beam SELECTION is driven by the RENDERED map X_hat; the RATE is always",
        "evaluated on the GROUND-TRUTH map X_gt. This mirrors the paper's",
        "protocol, in which CSI-RS measurement happens only on the subspace that",
        "the prior has already selected.",
        "",
        "SCALE CONVENTION",
        "----------------",
        "Raw per-location peak power spans about three orders of magnitude over",
        "the test set, and the renderer is trained against the per-location",
        "max-normalized target (utils/loss.py::normalize_mag_map). Both X_hat and",
        "X_gt are therefore max-normalized per location before selection and rate",
        "evaluation, so the SNR below is a PEAK SNR relative to the strongest beam",
        "pair at each location, and it is identical for prediction and ground",
        "truth. The rate metric is not scale invariant, so mixing a normalized",
        "prediction with an un-normalized ground truth would place the two schemes",
        "at different points on the log curve.",
        "",
        "CONFIGURATION",
        "-------------",
        f"checkpoint              : {checkpoint_path}",
        f"run directory           : {run_name}",
        f"checkpoint iteration    : {int(checkpoint.get('iteration', -1))}",
        f"test locations used     : {num_locations}",
        f"zero-power skipped      : {skipped}",
        f"beam grid (Nr x Nt)     : {num_rx_beams} x {num_tx_beams}",
        f"receive budget L_r      : {num_rx}",
        f"peak SNR                : {arguments.snr_db} dB (linear {snr_linear:.6f})",
        f"L_t grid                : {budgets}",
        f"random draws / location : {RANDOM_DRAWS} (seed {RANDOM_SEED})",
        "",
        "NET RATE PRELOG",
        "---------------",
        f"tau/T_B = 0.5 / Nt = {tau_over_tb:.8f}",
        "net rate = rate * (1 - L_t * tau/T_B), chosen so that a full sweep",
        f"(L_t = Nt = {num_tx_beams}) costs exactly 50% of the beam-management",
        "interval and therefore shows a clearly visible penalty.",
        "",
        "SANITY CHECKS",
        "-------------",
        f"(i)   Genie >= MIMO-GS >= Random : "
        f"{'PASS' if not violations else 'violations listed below'}",
        f"(ii)  agreement at L_t = Nt      : "
        f"{'PASS' if all('(ii)' not in v for v in violations) else 'FAIL'}",
        f"(iii) genie non-decreasing in L_t: "
        f"{'PASS' if not monotonic_breaks else 'FAIL (see note)'}",
    ]
    readme_lines += [f"      VIOLATION {v}" for v in violations]
    readme_lines += [f"      (iii) decrease at {e}" for e in monotonic_breaks]
    if monotonic_breaks:
        readme_lines += [
            "",
            "      NOTE on (iii): the metric divides transmit power by L_t, so a",
            "      larger budget spreads P over more beams. Once the budget exceeds",
            "      the number of distinct paths the L_r receive beams can exploit,",
            "      the 1/L_t power split outweighs the extra collected energy and",
            "      the rate falls. This is a property of the specified metric, not",
            "      a selection bug; genie remains the per-L_t upper bound.",
        ]

    with open(os.path.join(output_dir, "README.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(readme_lines) + "\n")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    best_index = int(np.argmax(net_rates["MIMO-GS"]))

    print("")
    print("=" * 78)
    print("[eval_bm] SUMMARY")
    print("=" * 78)
    print(f"  {'L_t':>5}{'prelog':>9}{'MIMO-GS':>10}{'Genie':>10}{'Random':>10}"
          f"{'net GS':>10}{'align':>9}")
    for i, budget in enumerate(budgets):
        print(
            f"  {budget:>5}{prelog[i]:>9.4f}{mean_rates['MIMO-GS'][i]:>10.4f}"
            f"{mean_rates['Genie'][i]:>10.4f}{mean_rates['Random'][i]:>10.4f}"
            f"{net_rates['MIMO-GS'][i]:>10.4f}{alignment['MIMO-GS'][i]:>9.4f}"
        )
    print("")
    print(f"  full sweep (L_t={num_tx_beams}) rate : {full_sweep_rate:.4f} bit/s/Hz"
          f"  -> net {full_sweep_net:.4f}")
    print(f"  MIMO-GS net-rate optimum   : L_t={budgets[best_index]} "
          f"-> {net_rates['MIMO-GS'][best_index]:.4f} bit/s/Hz "
          f"({net_rates['MIMO-GS'][best_index] / max(full_sweep_net, EPS):.2f}x "
          f"the full sweep)")
    print("")
    print(f"[eval_bm] Outputs written to {output_dir}")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
