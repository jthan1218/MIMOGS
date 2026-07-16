"""
simul_power_capture2_rev.py
======

Beam-search time comparison for rendered long-term beamspace statistics.

Run from the MIMO-GS repo root:
    python simul_power_capture2_rev.py

Compared methods:
    1) 3D-GS Beam Search: probes beam pairs in descending rendered power order.
    2) Exhaustive Search: sweeps all beam pairs sequentially in a fixed row-major order.
    3) Random Search: sweeps beam pairs in random order, averaged over trials.

Metric:
    rho_k(p) = max true power among the first k probed beam pairs
               --------------------------------------------------
                         max true power over all beam pairs

Thus rho_k=1 means that the best beam pair has been reached by time k.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np

# Robust import when the script is executed from outside the repo root.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mimogs_eval_common import (  # noqa: E402
    DEFAULT_CKPT,
    common_cli_args,
    load_context,
    savefig_pdf_png,
    script_out_dir,
    setup_matplotlib,
    write_csv,
)

EPS = 1e-12


# -----------------------------------------------------------------------------
# Path / preprocessing helpers
# -----------------------------------------------------------------------------
def resolve_ckpt(path: str) -> str:
    """Resolve a checkpoint path relative to this script location."""
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def clean_nonnegative(x: np.ndarray) -> np.ndarray:
    """Return a finite, nonnegative float64 array."""
    y = np.asarray(x, dtype=np.float64)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(y, 0.0)


def predicted_descending_order(score_maps: np.ndarray) -> np.ndarray:
    """
    Beam-pair order induced by the rendered statistic.

    score_maps: [N, Nr, Nt]
    return    : [N, Nr*Nt], descending order of flattened beam-pair indices.
    """
    score = clean_nonnegative(score_maps)
    flat = score.reshape(score.shape[0], -1)
    return np.argsort(-flat, axis=1, kind="mergesort").astype(np.int64)


def exhaustive_sweep_order(n_samples: int, n_beams: int) -> np.ndarray:
    """
    Fixed exhaustive beam sweeping order.

    This is not a single endpoint. It produces the same type of running
    best-so-far curve as the other methods, while eventually sweeping all beams.
    """
    base = np.arange(n_beams, dtype=np.int64)
    return np.tile(base.reshape(1, -1), (int(n_samples), 1))


# -----------------------------------------------------------------------------
# Core metric
# -----------------------------------------------------------------------------
def cumulative_power_ratio(
    true_power_flat: np.ndarray,
    order: np.ndarray,
    target_ratio: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cumulative captured beamspace power ratio for a given sweeping order.

    rho_k(p) = cumulative true power among the first k swept beam pairs
               ------------------------------------------------------
                       total true power over all beam pairs
    """
    power = clean_nonnegative(true_power_flat)
    order = np.asarray(order, dtype=np.int64)

    if power.ndim != 2:
        raise ValueError(f"true_power_flat must be 2-D, got shape {power.shape}")
    if order.shape != power.shape:
        raise ValueError(f"order shape {order.shape} does not match power shape {power.shape}")

    selected = np.take_along_axis(power, order, axis=1)
    cumulative = np.cumsum(selected, axis=1)

    total_power = np.maximum(np.sum(power, axis=1, keepdims=True), EPS)
    ratio = np.clip(cumulative / total_power, 0.0, 1.0)

    target_ratio = float(np.clip(target_ratio, 0.0, 1.0))
    hit = ratio >= (target_ratio - 1e-12)
    hit_time = np.where(hit.any(axis=1), np.argmax(hit, axis=1) + 1, power.shape[1])

    return ratio.mean(axis=0), ratio, hit_time.astype(np.int64)


def random_search_stats(
    true_power_flat: np.ndarray,
    n_trials: int,
    seed: int,
    target_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo random beam-sweeping curve and ratio-one hit times."""
    power = clean_nonnegative(true_power_flat)
    n_samples, n_beams = power.shape
    n_trials = max(1, int(n_trials))

    rng = np.random.default_rng(int(seed))
    curve_acc = np.zeros(n_beams, dtype=np.float64)
    hit_times = np.empty((n_trials, n_samples), dtype=np.int64)

    for t in range(n_trials):
        # Independent random beam order for each UE location.
        random_key = rng.random((n_samples, n_beams))
        order = np.argsort(random_key, axis=1).astype(np.int64)
        curve, _, h = cumulative_power_ratio(power, order, target_ratio=target_ratio)
        curve_acc += curve
        hit_times[t] = h

    return curve_acc / float(n_trials), hit_times.reshape(-1)


def choose_eval_indices(
    true_power_flat: np.ndarray,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    """Choose nonzero-power test UE locations. Default use is 100 samples."""
    power = clean_nonnegative(true_power_flat)
    valid = np.flatnonzero(np.max(power, axis=1) > EPS)
    if valid.size == 0:
        raise ValueError("No nonzero-power test UE samples were found.")

    num_samples = int(num_samples)
    if num_samples <= 0 or num_samples >= valid.size:
        return valid

    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(valid, size=num_samples, replace=False))


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def evaluate(
    ckpt_path: str,
    use_cache: bool,
    device: Optional[str],
    num_samples: int,
    random_trials: int,
    seed: int,
    target_ratio: float,
) -> Dict[str, object]:
    """Evaluate 3D-GS, exhaustive, and random beam sweeping."""
    ctx = load_context(ckpt_path=ckpt_path, use_cache=use_cache, device=device, verbose=True)

    # ctx.g_true is the scaled true beam-pair power map. The global scale cancels
    # in the ratio, but using g_true keeps this consistent with the common eval code.
    true_power = clean_nonnegative(ctx.g_true)
    n_total, Nr, Nt = true_power.shape
    n_beams = Nr * Nt
    true_power_flat_all = true_power.reshape(n_total, n_beams)

    eval_idx = choose_eval_indices(
        true_power_flat_all,
        num_samples=num_samples,
        seed=seed,
    )

    true_power_flat = true_power_flat_all[eval_idx]
    pred_score_maps = clean_nonnegative(ctx.Mhat)[eval_idx]

    gs_order = predicted_descending_order(pred_score_maps)

    ex_order = exhaustive_sweep_order(n_samples=true_power_flat.shape[0], n_beams=n_beams)

    gs_curve, _, gs_hit = cumulative_power_ratio(
        true_power_flat,
        gs_order,
        target_ratio=target_ratio,
    )

    ex_curve, _, ex_hit = cumulative_power_ratio(
        true_power_flat,
        ex_order,
        target_ratio=target_ratio,
    )

    rand_curve, rand_hit = random_search_stats(
        true_power_flat=true_power_flat,
        n_trials=random_trials,
        seed=seed,
        target_ratio=target_ratio,
    )

    x = np.arange(1, n_beams + 1, dtype=np.int64)
    return {
        "x_beams": x,
        "curves": {
            "3D-GS Beam Search": gs_curve,
            "Exhaustive Search": ex_curve,
            "Random Search": rand_curve,
        },
        "hit_times": {
            "3D-GS Beam Search": gs_hit,
            "Exhaustive Search": ex_hit,
            "Random Search": rand_hit,
        },
        "Nr": int(Nr),
        "Nt": int(Nt),
        "n_beams": int(n_beams),
        "n_total": int(n_total),
        "n_valid": int(np.sum(np.max(true_power_flat_all, axis=1) > EPS)),
        "n_eval": int(eval_idx.size),
        "eval_indices": eval_idx.astype(np.int64),
        "random_trials": int(max(1, random_trials)),
    }


# -----------------------------------------------------------------------------
# Plot / CSV output
# -----------------------------------------------------------------------------
def summarize_hits(hit_times: np.ndarray, Lp: int) -> Dict[str, float]:
    """Summary statistics for ratio-one hit time."""
    h = np.asarray(hit_times, dtype=np.float64)
    Lp = int(Lp)
    return {
        "mean_probes": float(np.mean(h)),
        "median_probes": float(np.median(h)),
        "p10_probes": float(np.percentile(h, 10)),
        "p90_probes": float(np.percentile(h, 90)),
        "mean_time": float(np.mean(h) * Lp),
        "median_time": float(np.median(h) * Lp),
        "p10_time": float(np.percentile(h, 10) * Lp),
        "p90_time": float(np.percentile(h, 90) * Lp),
    }


def plot_search_power_ratio(
    x_beams: np.ndarray,
    curves: Dict[str, np.ndarray],
    Lp: int,
    out_dir: str,
) -> Tuple[str, str, str]:
    """Plot running best-beam power ratio versus search time."""
    plt = setup_matplotlib()
    names = ["3D-GS Beam Search", "Exhaustive Search", "Random Search"]
    x_time = np.asarray(x_beams, dtype=np.float64) * float(Lp)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name in names:
        ax.plot(x_time, curves[name], linewidth=2.0, label=name)

    ax.set_title("Search Power Ratio")
    ax.set_xlabel("Search time (pilot symbols)")
    ax.set_ylabel("Best-beam power ratio")
    ax.set_xlim(float(x_time[0]), float(x_time[-1]))
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=True)
    fig.tight_layout()

    pdf_path, png_path = savefig_pdf_png(fig, out_dir, "search_power_ratio")
    plt.close(fig)

    rows = []
    for i, k in enumerate(x_beams):
        rows.append([
            int(k),
            int(k * int(Lp)),
            float(curves["3D-GS Beam Search"][i]),
            float(curves["Exhaustive Search"][i]),
            float(curves["Random Search"][i]),
        ])

    csv_path = write_csv(
        os.path.join(out_dir, "search_power_ratio.csv"),
        [
            "probed_beam_pairs",
            "search_time_pilot_symbols",
            "3D-GS Beam Search",
            "Exhaustive Search",
            "Random Search",
        ],
        rows,
    )
    return pdf_path, png_path, csv_path


def plot_ratio_one_time(
    hit_times: Dict[str, np.ndarray],
    Lp: int,
    out_dir: str,
) -> Tuple[str, str, str]:
    """Plot mean time to reach best-beam power ratio one."""
    plt = setup_matplotlib()
    names = ["3D-GS Beam Search", "Exhaustive Search", "Random Search"]
    stats = {name: summarize_hits(hit_times[name], Lp=Lp) for name in names}

    means = np.asarray([stats[name]["mean_time"] for name in names], dtype=np.float64)
    medians = np.asarray([stats[name]["median_time"] for name in names], dtype=np.float64)
    p10 = np.asarray([stats[name]["p10_time"] for name in names], dtype=np.float64)
    p90 = np.asarray([stats[name]["p90_time"] for name in names], dtype=np.float64)
    yerr = np.vstack([np.maximum(means - p10, 0.0), np.maximum(p90 - means, 0.0)])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=yerr, capsize=4)
    ax.scatter(x, medians, marker="D", s=34, label="Median")
    ax.set_title("Cumulative Power Ratio")
    ax.set_xlabel("Beam sweeping symbols")
    ax.set_ylabel("Cumulative power ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.legend(frameon=True)
    fig.tight_layout()

    pdf_path, png_path = savefig_pdf_png(fig, out_dir, "ratio_one_time")
    plt.close(fig)

    rows = []
    for name in names:
        s = stats[name]
        rows.append([
            name,
            s["mean_probes"],
            s["median_probes"],
            s["p10_probes"],
            s["p90_probes"],
            s["mean_time"],
            s["median_time"],
            s["p10_time"],
            s["p90_time"],
        ])

    csv_path = write_csv(
        os.path.join(out_dir, "ratio_one_time.csv"),
        [
            "method",
            "mean_probes",
            "median_probes",
            "p10_probes",
            "p90_probes",
            "mean_time_pilot_symbols",
            "median_time_pilot_symbols",
            "p10_time_pilot_symbols",
            "p90_time_pilot_symbols",
        ],
        rows,
    )
    return pdf_path, png_path, csv_path


def write_eval_indices(eval_indices: np.ndarray, out_dir: str) -> str:
    """Save the evaluated test indices for reproducibility."""
    rows = [[int(i)] for i in np.asarray(eval_indices).reshape(-1)]
    return write_csv(os.path.join(out_dir, "eval_indices.csv"), ["test_index"], rows)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare 3D-GS, exhaustive, and random beam search time."
    )
    common_cli_args(parser)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of nonzero-power test UE locations to evaluate. Use 0 for all.",
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=300,
        help="Monte-Carlo trials for Random Search.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for UE subsampling and Random Search.",
    )
    parser.add_argument(
    "--target-ratio",
    type=float,
    default=0.99,
    help="Target cumulative power ratio for the sweeping-time summary.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ckpt_path = resolve_ckpt(args.ckpt if args.ckpt else DEFAULT_CKPT)
    out_dir = script_out_dir(ckpt_path, "simul_power_capture2")

    result = evaluate(
        ckpt_path=ckpt_path,
        use_cache=not args.no_cache,
        device=args.device,
        num_samples=args.num_samples,
        random_trials=args.random_trials,
        seed=args.seed,
        target_ratio=args.target_ratio,
    )

    ratio_pdf, ratio_png, ratio_csv = plot_search_power_ratio(
        x_beams=result["x_beams"],
        curves=result["curves"],
        Lp=int(args.Lp),
        out_dir=out_dir,
    )
    time_pdf, time_png, time_csv = plot_ratio_one_time(
        hit_times=result["hit_times"],
        Lp=int(args.Lp),
        out_dir=out_dir,
    )
    idx_csv = write_eval_indices(result["eval_indices"], out_dir=out_dir)

    print("[simul_power_capture2] Done")
    print(f"  checkpoint    : {ckpt_path}")
    print(f"  beamspace     : {result['Nr']} x {result['Nt']} ({result['n_beams']} beam pairs)")
    print(f"  total test    : {result['n_total']}")
    print(f"  valid test    : {result['n_valid']}")
    print(f"  eval samples  : {result['n_eval']}")
    print(f"  random trials : {result['random_trials']}")
    print(f"  ratio png     : {ratio_png}")
    print(f"  ratio pdf     : {ratio_pdf}")
    print(f"  ratio csv     : {ratio_csv}")
    print(f"  time png      : {time_png}")
    print(f"  time pdf      : {time_pdf}")
    print(f"  time csv      : {time_csv}")
    print(f"  indices csv   : {idx_csv}")


if __name__ == "__main__":
    main()
