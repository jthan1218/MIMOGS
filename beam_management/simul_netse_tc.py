#!/usr/bin/env python
"""
simul_netse_tc.py  --  MAIN figure: net spectral efficiency vs coherence length T_c
===================================================================================

Standalone runnable:
    python simul_netse_tc.py
    python simul_netse_tc.py --SNR_dB 10 --no-cache
    python simul_netse_tc.py --Tc_min 80 --Tc_max 4000 --num_Tc 40

Story
-----
As the coherence length T_c grows, probing overhead K*Lp/Tc becomes cheap, so MIMO-GS
can afford a larger candidate set: rho_bar_align -> 1 and the prelog -> 1, hence the net
SE R_bar_eff -> R_genie (the zero-overhead ceiling). The exhaustive full sweep pays a
fixed 64*Lp overhead and therefore lags far behind, only slowly approaching the ceiling.

For every T_c, MIMO-GS adapts its margin: we sweep Delta and pick Delta*(T_c) that
maximizes R_bar_eff -> the adaptive MIMO-GS curve. We plot three curves vs T_c:
    - Genie ceiling (flat, zero overhead)
    - MIMO-GS (adaptive Delta*)
    - Exhaustive sweep (K = Nr*Nt = 64; omitted where 64*Lp >= Tc)

Outputs (under outputs/<run>/beam_eval/simul_netse_tc/):
    netse_vs_tc.pdf / .png
    netse_vs_tc.csv   with (T_c, R_bar_eff_mimogs, K_bar, Delta_star, R_exhaustive, R_genie)
"""

import argparse
import numpy as np

import mimogs_eval_common as C


def parse_args():
    p = argparse.ArgumentParser(description="Net SE vs coherence length T_c (MAIN figure).")
    C.common_cli_args(p)
    p.add_argument("--SNR_dB", type=float, default=C.DEFAULTS.SNR_dB,
                   help="Fixed operating SNR in dB.")
    p.add_argument("--Tc_min", type=float, default=None,
                   help="Smallest T_c (default: just above Nr*Nt*Lp = 64*Lp).")
    p.add_argument("--Tc_max", type=float, default=4096.0, help="Largest T_c.")
    p.add_argument("--num_Tc", type=int, default=40, help="Number of T_c grid points (log-spaced).")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = C.script_out_dir(args.ckpt, "simul_netse_tc")

    ctx = C.load_context(ckpt_path=args.ckpt, use_cache=not args.no_cache, device=args.device)

    SNR_lin = C.db2lin(args.SNR_dB)
    Lp = args.Lp
    NrNt = ctx.Nr * ctx.Nt  # 64

    # T_c grid: from just above the exhaustive overhead up to Tc_max (log-spaced so the
    # convergence toward the ceiling is visible).
    tc_min = args.Tc_min if args.Tc_min is not None else (NrNt * Lp + 1)
    Tc_grid = np.unique(np.round(
        np.logspace(np.log10(tc_min), np.log10(args.Tc_max), args.num_Tc)
    )).astype(float)

    delta_grid = C.default_delta_grid()

    # Genie ceiling is independent of T_c.
    R_genie_bar = float(np.mean(C.R_genie(ctx.g_star, SNR_lin)))

    rows = []
    mimogs_R, mimogs_K, mimogs_D, exh_R = [], [], [], []

    print(f"[netse_tc] SNR={args.SNR_dB} dB, Lp={Lp}, Nr*Nt={NrNt}, "
          f"gain_scale={ctx.gain_scale:.4g} (median g_star -> 0 dB)")

    for Tc in Tc_grid:
        bd = C.best_delta(ctx, SNR_lin, Tc=Tc, Lp=Lp, delta_grid=delta_grid)
        R_exh = float(np.nanmean(C.R_exhaustive(ctx.g_star, SNR_lin, ctx.Nr, ctx.Nt, Lp=Lp, Tc=Tc)))

        mimogs_R.append(bd["R_bar_eff"])
        mimogs_K.append(bd["K_bar"])
        mimogs_D.append(bd["Delta"])
        exh_R.append(R_exh)

        rows.append([f"{Tc:.1f}", f"{bd['R_bar_eff']:.6f}", f"{bd['K_bar']:.4f}",
                     f"{bd['Delta']:.4f}",
                     ("" if not np.isfinite(R_exh) else f"{R_exh:.6f}"),
                     f"{R_genie_bar:.6f}"])

    mimogs_R = np.array(mimogs_R)
    mimogs_K = np.array(mimogs_K)
    mimogs_D = np.array(mimogs_D)
    exh_R = np.array(exh_R)

    # ---- CSV ----
    csv_path = C.write_csv(
        f"{out_dir}/netse_vs_tc.csv",
        ["T_c", "R_bar_eff_mimogs", "K_bar", "Delta_star", "R_exhaustive", "R_genie"],
        rows,
    )

    # ---- Figure ----
    plt = C.setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.axhline(R_genie_bar, color="k", ls="--", lw=1.8,
               label=f"Genie ceiling (no overhead) = {R_genie_bar:.3f}")
    ax.plot(Tc_grid, mimogs_R, "-o", color="tab:blue", ms=4, lw=2.0,
            label=r"MIMO-GS (adaptive $\Delta^\star$)")

    exh_finite = np.isfinite(exh_R)
    ax.plot(Tc_grid[exh_finite], exh_R[exh_finite], "-s", color="tab:red", ms=4, lw=1.8,
            label=r"Exhaustive sweep ($K=N_rN_t=64$)")

    ax.set_xscale("log")
    ax.set_xlabel(r"Coherence length $T_c$ (symbols)")
    ax.set_ylabel(r"Net spectral efficiency $\bar{R}_{\rm eff}$ (bits/s/Hz)")
    ax.set_title(rf"Net SE vs $T_c$  (SNR = {args.SNR_dB:.0f} dB)")
    ax.legend(loc="lower right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    pdf, png = C.savefig_pdf_png(fig, out_dir, "netse_vs_tc")

    # ---- stdout summary ----
    print("\n========== simul_netse_tc summary ==========")
    print(f"SNR = {args.SNR_dB:.1f} dB,  Lp = {Lp},  Nr*Nt = {NrNt}")
    print(f"Genie ceiling (flat)      : {R_genie_bar:.4f} bits/s/Hz")
    i_lo = 0
    i_hi = len(Tc_grid) - 1
    print(f"MIMO-GS @ Tc={Tc_grid[i_lo]:.0f}     : R={mimogs_R[i_lo]:.4f}, "
          f"Delta*={mimogs_D[i_lo]:.2f}, K_bar={mimogs_K[i_lo]:.2f}")
    print(f"MIMO-GS @ Tc={Tc_grid[i_hi]:.0f}   : R={mimogs_R[i_hi]:.4f}, "
          f"Delta*={mimogs_D[i_hi]:.2f}, K_bar={mimogs_K[i_hi]:.2f} "
          f"({100*mimogs_R[i_hi]/R_genie_bar:.1f}% of genie)")
    if np.any(exh_finite):
        print(f"Exhaustive @ Tc={Tc_grid[i_hi]:.0f} : R={exh_R[i_hi]:.4f}")
        gain = mimogs_R[i_hi] / exh_R[i_hi] if exh_R[i_hi] > 0 else float('nan')
        print(f"MIMO-GS / Exhaustive @ Tc={Tc_grid[i_hi]:.0f} : x{gain:.2f}")
    print(f"CSV : {csv_path}")
    print(f"FIG : {pdf}\n      {png}")


if __name__ == "__main__":
    main()
