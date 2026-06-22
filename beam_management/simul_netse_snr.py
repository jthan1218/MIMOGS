#!/usr/bin/env python
"""
simul_netse_snr.py  --  net SE vs SNR, and net SE vs Delta (find Delta*)
========================================================================

Standalone runnable:
    python simul_netse_snr.py
    python simul_netse_snr.py --Tc 256 --snr_min -10 --snr_max 30 --snr_step 5
    python simul_netse_snr.py --delta_snr_dB 10 --no-cache

Figure 1 (net SE vs SNR):
    Fix T_c. Sweep SNR. Plot R_bar_eff for
        - MIMO-GS (using Delta* found at this T_c, at a reference SNR),
        - Exhaustive sweep (K = Nr*Nt),
        - Genie bound.

Figure 2 (net SE vs Delta):
    At a fixed mid SNR and fixed T_c, plot R_bar_eff vs Delta over [0,1] and mark the
    interior optimum Delta*. This shows the optimum beats both the greedy K=1 choice
    (Delta=0) and the full sweep (Delta=1 -> K=64). Prints Delta* and its
    (K_bar, rho_bar_align, R_bar_eff).

Outputs (under outputs/<run>/beam_eval/simul_netse_snr/):
    netse_vs_snr.pdf/.png  + netse_vs_snr.csv
    netse_vs_delta.pdf/.png + netse_vs_delta.csv
"""

import argparse
import numpy as np

import mimogs_eval_common as C


def parse_args():
    p = argparse.ArgumentParser(description="Net SE vs SNR and vs Delta.")
    C.common_cli_args(p)
    p.add_argument("--Tc", type=float, default=C.DEFAULTS.Tc, help="Fixed coherence length.")
    p.add_argument("--snr_min", type=float, default=-10.0, help="Min SNR (dB).")
    p.add_argument("--snr_max", type=float, default=30.0, help="Max SNR (dB).")
    p.add_argument("--snr_step", type=float, default=5.0, help="SNR step (dB).")
    p.add_argument("--delta_snr_dB", type=float, default=C.DEFAULTS.SNR_dB,
                   help="SNR (dB) used for the net-SE-vs-Delta figure and for picking Delta*.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = C.script_out_dir(args.ckpt, "simul_netse_snr")
    ctx = C.load_context(ckpt_path=args.ckpt, use_cache=not args.no_cache, device=args.device)

    Lp = args.Lp
    Tc = args.Tc
    NrNt = ctx.Nr * ctx.Nt
    delta_grid = C.default_delta_grid()

    print(f"[netse_snr] Tc={Tc}, Lp={Lp}, Nr*Nt={NrNt}, "
          f"gain_scale={ctx.gain_scale:.4g} (median g_star -> 0 dB)")

    # ========================================================================
    # Figure 2 first: net SE vs Delta at the reference SNR -> gives Delta*
    # ========================================================================
    snr_ref = C.db2lin(args.delta_snr_dB)

    delta_R, delta_K, delta_rho = [], [], []
    for D in delta_grid:
        res = C.evaluate_delta(ctx, float(D), snr_ref, Lp=Lp, Tc=Tc)
        delta_R.append(res["R_bar_eff"])
        delta_K.append(res["K_bar"])
        delta_rho.append(res["rho_bar"])
    delta_R = np.array(delta_R)
    delta_K = np.array(delta_K)
    delta_rho = np.array(delta_rho)

    j_star = int(np.nanargmax(delta_R))
    Delta_star = float(delta_grid[j_star])
    R_star = float(delta_R[j_star])
    K_star = float(delta_K[j_star])
    rho_star = float(delta_rho[j_star])

    # CSV for delta sweep
    C.write_csv(
        f"{out_dir}/netse_vs_delta.csv",
        ["Delta", "R_bar_eff", "K_bar", "rho_bar_align"],
        [[f"{d:.4f}", f"{r:.6f}", f"{k:.4f}", f"{rho:.6f}"]
         for d, r, k, rho in zip(delta_grid, delta_R, delta_K, delta_rho)],
    )

    plt = C.setup_matplotlib()
    fig2, ax2 = plt.subplots(figsize=(7.2, 4.8))
    ax2.plot(delta_grid, delta_R, "-o", color="tab:blue", ms=3.5, lw=2.0, label=r"MIMO-GS $\bar{R}_{\rm eff}$")
    # reference endpoints
    ax2.axhline(delta_R[0], color="tab:green", ls=":", lw=1.5,
                label=rf"$\Delta=0$ (greedy $K=1$) = {delta_R[0]:.3f}")
    R_full = delta_R[-1]
    if np.isfinite(R_full):
        ax2.axhline(R_full, color="tab:red", ls=":", lw=1.5,
                    label=rf"$\Delta=1$ (full sweep) = {R_full:.3f}")
    ax2.plot([Delta_star], [R_star], "*", color="k", ms=16,
             label=rf"$\Delta^\star={Delta_star:.2f}$, $\bar R={R_star:.3f}$")
    ax2.set_xlabel(r"Candidate margin $\Delta$")
    ax2.set_ylabel(r"Net spectral efficiency $\bar{R}_{\rm eff}$ (bits/s/Hz)")
    ax2.set_title(rf"Net SE vs $\Delta$  (SNR={args.delta_snr_dB:.0f} dB, $T_c$={Tc:.0f})")
    ax2.legend(loc="lower center")
    fig2.tight_layout()
    pdf2, png2 = C.savefig_pdf_png(fig2, out_dir, "netse_vs_delta")

    # ========================================================================
    # Figure 1: net SE vs SNR, using the Delta* found above
    # ========================================================================
    snr_db_grid = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step)

    R_mimogs, R_exh, R_gen = [], [], []
    for snr_db in snr_db_grid:
        snr = C.db2lin(snr_db)
        res = C.evaluate_delta(ctx, Delta_star, snr, Lp=Lp, Tc=Tc)
        R_mimogs.append(res["R_bar_eff"])
        R_exh.append(float(np.nanmean(C.R_exhaustive(ctx.g_star, snr, ctx.Nr, ctx.Nt, Lp=Lp, Tc=Tc))))
        R_gen.append(float(np.mean(C.R_genie(ctx.g_star, snr))))

    R_mimogs = np.array(R_mimogs)
    R_exh = np.array(R_exh)
    R_gen = np.array(R_gen)

    C.write_csv(
        f"{out_dir}/netse_vs_snr.csv",
        ["SNR_dB", "R_bar_eff_mimogs", "R_exhaustive", "R_genie", "Delta_star_used"],
        [[f"{s:.2f}", f"{rm:.6f}",
          ("" if not np.isfinite(re) else f"{re:.6f}"),
          f"{rg:.6f}", f"{Delta_star:.4f}"]
         for s, rm, re, rg in zip(snr_db_grid, R_mimogs, R_exh, R_gen)],
    )

    fig1, ax1 = plt.subplots(figsize=(7.2, 4.8))
    ax1.plot(snr_db_grid, R_gen, "--", color="k", lw=1.8, label="Genie bound (no overhead)")
    ax1.plot(snr_db_grid, R_mimogs, "-o", color="tab:blue", ms=4, lw=2.0,
             label=rf"MIMO-GS ($\Delta^\star={Delta_star:.2f}$)")
    fin = np.isfinite(R_exh)
    ax1.plot(snr_db_grid[fin], R_exh[fin], "-s", color="tab:red", ms=4, lw=1.8,
             label=r"Exhaustive sweep ($K=64$)")
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel(r"Net spectral efficiency $\bar{R}_{\rm eff}$ (bits/s/Hz)")
    ax1.set_title(rf"Net SE vs SNR  ($T_c$={Tc:.0f}, $L_p$={Lp})")
    ax1.legend(loc="upper left")
    fig1.tight_layout()
    pdf1, png1 = C.savefig_pdf_png(fig1, out_dir, "netse_vs_snr")

    # ---- stdout summary ----
    print("\n========== simul_netse_snr summary ==========")
    print(f"Tc = {Tc:.0f},  Lp = {Lp},  Nr*Nt = {NrNt}")
    print(f"[Delta sweep @ {args.delta_snr_dB:.0f} dB]  Delta* = {Delta_star:.3f}")
    print(f"   K_bar = {K_star:.3f},  rho_bar_align = {rho_star:.4f},  R_bar_eff = {R_star:.4f}")
    print(f"   vs greedy K=1 (Delta=0): R = {delta_R[0]:.4f}")
    print(f"   vs full sweep (Delta=1): R = {delta_R[-1]:.4f}")
    # headline at the reference SNR
    k_ref = int(np.argmin(np.abs(snr_db_grid - args.delta_snr_dB)))
    print(f"[SNR sweep @ {snr_db_grid[k_ref]:.0f} dB]  "
          f"MIMO-GS={R_mimogs[k_ref]:.4f}, Exhaustive={R_exh[k_ref]:.4f}, Genie={R_gen[k_ref]:.4f}")
    if np.isfinite(R_exh[k_ref]) and R_exh[k_ref] > 0:
        print(f"   MIMO-GS / Exhaustive = x{R_mimogs[k_ref]/R_exh[k_ref]:.2f}, "
              f"MIMO-GS / Genie = {100*R_mimogs[k_ref]/R_gen[k_ref]:.1f}%")
    print(f"CSV : {out_dir}/netse_vs_snr.csv\n      {out_dir}/netse_vs_delta.csv")
    print(f"FIG : {pdf1}\n      {pdf2}")


if __name__ == "__main__":
    main()
