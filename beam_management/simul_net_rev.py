#!/usr/bin/env python
"""
simul_net_rev.py
================

Net achievable rate versus coherence length T_c.

Run from the MIMO-GS repo root:
    python simul_net_rev.py

Outputs:
    outputs/<run>/beam_eval/simul_net_rev/net_rate_vs_tc.png
    outputs/<run>/beam_eval/simul_net_rev/net_rate_vs_tc.pdf
    outputs/<run>/beam_eval/simul_net_rev/net_rate_vs_tc.csv

Metric:
    R_eff(p) = (1 - K(p)L_p/T_c) log2(1 + SNR * rho_align(p) * g_star(p)).

The selected beam pair is treated as a scalar effective channel, so the rate term
is the SISO capacity expression after beam alignment. The prelog term accounts
for beam-sweeping overhead.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from typing import Optional

import numpy as np

# Make the script runnable even when launched from outside the repo root.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import mimogs_eval_common as C  # noqa: E402

EPS = 1e-12


def resolve_path(path: str) -> str:
    """Resolve a path relative to this script location."""
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Net achievable rate versus coherence length T_c.")
    C.common_cli_args(p)
    p.add_argument("--SNR_dB", type=float, default=C.DEFAULTS.SNR_dB,
                   help="Operating SNR in dB.")
    p.add_argument("--Tc_min", type=float, default=None,
                   help="Smallest T_c. Default: Nr*Nt*Lp + 1.")
    p.add_argument("--Tc_max", type=float, default=4096.0,
                   help="Largest T_c.")
    p.add_argument("--num_Tc", type=int, default=40,
                   help="Number of log-spaced T_c samples.")
    p.add_argument("--num_delta", type=int, default=41,
                   help="Number of Delta values for adaptive MIMO-GS search.")
    p.add_argument("--num-samples", type=int, default=0,
                   help="Number of valid test UE locations to average. 0 uses all valid locations.")
    p.add_argument("--seed", type=int, default=7,
                   help="Random seed used when --num-samples is positive.")
    p.add_argument("--no-equation", action="store_true",
                   help="Do not print the metric equation above the plot.")
    return p.parse_args()


def clean_nonnegative(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(y, 0.0)


def choose_eval_indices(ctx: C.EvalContext, num_samples: int, seed: int) -> np.ndarray:
    """Choose nonzero-power test locations. If num_samples <= 0, use all valid locations."""
    g_flat = clean_nonnegative(ctx.g_true).reshape(ctx.ntest, -1)
    valid = np.flatnonzero(np.max(g_flat, axis=1) > EPS)
    if valid.size == 0:
        raise ValueError("No nonzero-power test UE locations were found.")

    num_samples = int(num_samples)
    if num_samples <= 0 or num_samples >= valid.size:
        return valid.astype(np.int64)

    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(valid, size=num_samples, replace=False)).astype(np.int64)


def subset_context(ctx: C.EvalContext, idx: np.ndarray) -> C.EvalContext:
    """Return an EvalContext restricted to selected UE locations."""
    idx = np.asarray(idx, dtype=np.int64)
    return replace(
        ctx,
        Mhat=ctx.Mhat[idx],
        Mtrue=ctx.Mtrue[idx],
        g_true=ctx.g_true[idx],
        g_star=ctx.g_star[idx],
    )


def make_tc_grid(tc_min: float, tc_max: float, num_tc: int) -> np.ndarray:
    tc_min = float(tc_min)
    tc_max = float(tc_max)
    num_tc = int(max(2, num_tc))
    if tc_min <= 0.0:
        raise ValueError(f"Tc_min must be positive, got {tc_min}.")
    if tc_max <= tc_min:
        raise ValueError(f"Tc_max must be larger than Tc_min, got Tc_min={tc_min}, Tc_max={tc_max}.")

    grid = np.logspace(np.log10(tc_min), np.log10(tc_max), num_tc)
    grid = np.unique(np.round(grid).astype(np.float64))
    return grid


def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if not np.any(np.isfinite(x)):
        return float("nan")
    return float(np.nanmean(x))


def main() -> None:
    args = parse_args()
    ckpt_path = resolve_path(args.ckpt)
    out_dir = C.script_out_dir(ckpt_path, "simul_net_rev")

    ctx_all = C.load_context(
        ckpt_path=ckpt_path,
        use_cache=not args.no_cache,
        device=args.device,
        verbose=True,
    )

    eval_idx = choose_eval_indices(ctx_all, num_samples=args.num_samples, seed=args.seed)
    ctx = subset_context(ctx_all, eval_idx)

    SNR_lin = float(C.db2lin(args.SNR_dB))
    Lp = int(args.Lp)
    NrNt = int(ctx.Nr * ctx.Nt)

    tc_min = args.Tc_min if args.Tc_min is not None else (NrNt * Lp + 1.0)
    Tc_grid = make_tc_grid(tc_min=tc_min, tc_max=args.Tc_max, num_tc=args.num_Tc)
    delta_grid = C.default_delta_grid(num=int(args.num_delta))

    R_genie_bar = safe_mean(C.R_genie(ctx.g_star, SNR_lin))

    rows = []
    mimogs_R = []
    mimogs_K = []
    mimogs_D = []
    mimogs_rho = []
    exhaustive_R = []

    print(
        f"[simul_net_rev] SNR={args.SNR_dB:.1f} dB, Lp={Lp}, Nr*Nt={NrNt}, "
        f"samples={ctx.ntest}/{ctx_all.ntest}, gain_scale={ctx.gain_scale:.4g}"
    )

    for Tc in Tc_grid:
        # MIMO-GS: choose Delta*(T_c) maximizing average net achievable rate.
        bd = C.best_delta(ctx, SNR_lin, Tc=float(Tc), Lp=Lp, delta_grid=delta_grid)

        # Exhaustive sweep: perfect beam alignment after probing all Nr*Nt beam pairs.
        R_exh = safe_mean(C.R_exhaustive(ctx.g_star, SNR_lin, ctx.Nr, ctx.Nt, Lp=Lp, Tc=float(Tc)))

        mimogs_R.append(float(bd["R_bar_eff"]))
        mimogs_K.append(float(bd["K_bar"]))
        mimogs_D.append(float(bd["Delta"]))
        mimogs_rho.append(float(bd["rho_bar"]))
        exhaustive_R.append(R_exh)

        rows.append([
            f"{Tc:.1f}",
            f"{bd['R_bar_eff']:.8f}",
            f"{bd['K_bar']:.6f}",
            f"{bd['Delta']:.6f}",
            f"{bd['rho_bar']:.8f}",
            "" if not np.isfinite(R_exh) else f"{R_exh:.8f}",
            f"{R_genie_bar:.8f}",
            str(ctx.ntest),
        ])

    mimogs_R = np.asarray(mimogs_R, dtype=np.float64)
    mimogs_K = np.asarray(mimogs_K, dtype=np.float64)
    mimogs_D = np.asarray(mimogs_D, dtype=np.float64)
    mimogs_rho = np.asarray(mimogs_rho, dtype=np.float64)
    exhaustive_R = np.asarray(exhaustive_R, dtype=np.float64)

    csv_path = C.write_csv(
        os.path.join(out_dir, "net_rate_vs_tc.csv"),
        [
            "T_c",
            "R_bar_eff_mimogs",
            "K_bar",
            "Delta_star",
            "rho_align_bar",
            "R_exhaustive",
            "R_genie_no_overhead",
            "num_eval_locations",
        ],
        rows,
    )
    C.write_csv(
        os.path.join(out_dir, "eval_indices.csv"),
        ["test_index"],
        [[int(i)] for i in eval_idx],
    )

    plt = C.setup_matplotlib()
    if args.no_equation:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
    else:
        fig, ax = plt.subplots(figsize=(7.2, 5.35))

    ax.axhline(
        R_genie_bar,
        color="k",
        linestyle="--",
        linewidth=1.8,
        label=f"Genie ceiling (no overhead) = {R_genie_bar:.3f}",
    )
    ax.plot(
        Tc_grid,
        mimogs_R,
        "-o",
        markersize=4,
        linewidth=2.0,
        label=r"MIMO-GS (adaptive $\Delta^\star$)",
    )

    finite_exh = np.isfinite(exhaustive_R)
    if np.any(finite_exh):
        ax.plot(
            Tc_grid[finite_exh],
            exhaustive_R[finite_exh],
            "-s",
            markersize=4,
            linewidth=1.8,
            label=rf"Exhaustive sweep ($K=N_rN_t={NrNt}$)",
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"Coherence length $T_c$ (symbols)")
    ax.set_ylabel(r"Net achievable rate $\bar{R}_{\rm eff}$ (bits/s/Hz)")
    ax.set_title(rf"Net Achievable Rate vs $T_c$ (SNR = {args.SNR_dB:.0f} dB)")
    ax.legend(loc="lower right")
    ax.set_ylim(bottom=0.0)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90] if not args.no_equation else None)

    pdf_path, png_path = C.savefig_pdf_png(fig, out_dir, "net_rate_vs_tc")

    print("\n========== simul_net_rev summary ==========")
    print(f"SNR = {args.SNR_dB:.1f} dB, Lp = {Lp}, Nr*Nt = {NrNt}")
    print(f"Evaluation samples        : {ctx.ntest} / {ctx_all.ntest}")
    print(f"Genie ceiling             : {R_genie_bar:.4f} bits/s/Hz")
    print(
        f"MIMO-GS @ Tc={Tc_grid[0]:.0f}        : R={mimogs_R[0]:.4f}, "
        f"Delta*={mimogs_D[0]:.3f}, K_bar={mimogs_K[0]:.2f}, rho_bar={mimogs_rho[0]:.4f}"
    )
    print(
        f"MIMO-GS @ Tc={Tc_grid[-1]:.0f}      : R={mimogs_R[-1]:.4f}, "
        f"Delta*={mimogs_D[-1]:.3f}, K_bar={mimogs_K[-1]:.2f}, rho_bar={mimogs_rho[-1]:.4f}"
    )
    if np.any(finite_exh):
        last = np.flatnonzero(finite_exh)[-1]
        print(f"Exhaustive @ Tc={Tc_grid[last]:.0f}    : R={exhaustive_R[last]:.4f}")
        if exhaustive_R[last] > 0:
            print(f"MIMO-GS / Exhaustive      : x{mimogs_R[last] / exhaustive_R[last]:.2f}")
    print(f"CSV : {csv_path}")
    print(f"FIG : {pdf_path}\n      {png_path}")


if __name__ == "__main__":
    main()
