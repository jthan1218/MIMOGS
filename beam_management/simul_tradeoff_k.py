#!/usr/bin/env python
"""
simul_tradeoff_k.py  --  net SE vs Top-K probing budget (simple single-panel figure)
====================================================================================

Standalone runnable:
    python simul_tradeoff_k.py
    python simul_tradeoff_k.py --snr 10 --Tc 256
    python simul_tradeoff_k.py --Tc_list 128 256 1024 --no-cache

Idea
----
Fix the SNR and the coherence length Tc. Sweep the FIXED Top-K probing budget
K = 1, 2, ..., Nr*Nt (= 64). For every test location p we probe the K beam pairs with the
largest PREDICTED magnitude (Top-K from Mhat) and select the one with the largest TRUE
magnitude among them. The per-location net spectral efficiency is

    R_eff(K,p) = (1 - K*Lp/Tc) * log2( 1 + SNR * rho_align(K,p) * g*(p) ),

averaged over locations to give R_bar_eff(K). The genie ceiling (no overhead, perfect
alignment) is R_bar_genie = mean_p log2(1 + SNR * g*(p)).

K = 1   -> "Top-1 (prediction only)": trust the predicted best beam, probe just one pair.
K = 64  -> "Exhaustive sweep": probe every beam pair (rho_align = 1, maximum overhead).

Plotted on a LOG2 K-axis, R_bar_eff(K) reads as an inverted-U: it rises to a peak at a small
K* then declines toward the exhaustive point. The headline is that Top-1 (1 probe) sits far
ABOVE Exhaustive (64 probes): exhaustive wastes overhead while Top-1 already captures most of
the gain.

Everything (model loading, render + .npz cache, Top-K selection, alignment, net-SE, genie
reference, gain scaling) is REUSED from mimogs_eval_common.py; only the plotting is here.

Outputs (under outputs/<run>/beam_eval/simul_tradeoff_k/), one per Tc:
    netse_vs_k_Tc{Tc}.pdf / .png
    netse_vs_k_Tc{Tc}.csv   with columns [K, R_bar_eff, rho_bar_align]
"""

import argparse
import numpy as np

import mimogs_eval_common as C


def parse_args():
    p = argparse.ArgumentParser(
        description="Net SE vs Top-K probing budget (simple single-panel tradeoff figure).")
    C.common_cli_args(p)  # adds --ckpt, --no-cache, --Lp, --device
    p.add_argument("--snr", type=float, default=C.DEFAULTS.SNR_dB,
                   help="Fixed operating SNR in dB (default 10).")
    p.add_argument("--Tc", type=float, default=C.DEFAULTS.Tc,
                   help="Coherence length (default 256).")
    p.add_argument("--Tc_list", type=float, nargs="+", default=None,
                   help="Optional: emit one figure per Tc. Default is the single --Tc.")
    return p.parse_args()


def sweep_topk(ctx, K_grid, SNR_lin, Lp, Tc):
    """
    Sweep the fixed Top-K probing budget at one (SNR, Tc), reusing the shared helpers.

    For each K, evaluate_topk() runs topk_set (Top-K by predicted magnitude) + select_and_score
    (pick the largest TRUE magnitude among them) per location, and net_se() applies the
    (1 - K*Lp/Tc) overhead. Returns aligned arrays over K_grid:
        R_bar_eff : mean net SE over locations (NaN where K is infeasible)
        rho_bar   : mean alignment efficiency
        feasible  : bool, K*Lp < Tc
    """
    R_bar = np.full(len(K_grid), np.nan)
    rho_bar = np.zeros(len(K_grid))
    feasible = np.zeros(len(K_grid), dtype=bool)
    for i, K in enumerate(K_grid):
        res = C.evaluate_topk(ctx, int(K), SNR_lin, Lp=Lp, Tc=Tc)
        rho_bar[i] = res["rho_bar"]
        feasible[i] = bool(res["feasible"])
        if res["feasible"]:
            R_bar[i] = res["R_bar_eff"]
    return R_bar, rho_bar, feasible


def make_figure(plt, Tc, args, ctx, K_grid, R_bar, rho_bar, feasible,
                R_genie_bar, Lp, out_dir):
    """Build, annotate and save the single-panel net-SE-vs-K figure for one Tc."""
    NrNt = ctx.Nr * ctx.Nt
    Karr = np.array(K_grid, dtype=float)

    Kf = Karr[feasible]
    Rf = R_bar[feasible]

    # net-SE optimum K* (sweet spot).
    j = int(np.nanargmax(Rf))
    K_star = int(Kf[j])
    R_star = float(Rf[j])

    # the two headline operating points
    R_top1 = float(R_bar[0])                       # K = 1
    R_exh = float(R_bar[NrNt - 1]) if feasible[NrNt - 1] else float("nan")

    fig, ax = plt.subplots(figsize=(7.4, 4.9))

    # genie ceiling
    ax.axhline(R_genie_bar, color="k", ls="--", lw=1.6,
               label=f"Genie (no overhead) = {R_genie_bar:.3f}")

    # main curve: MIMO-GS Top-K net SE (inverted-U on log2 axis)
    ax.plot(Kf, Rf, "-", color="tab:blue", lw=2.2, marker="o", ms=4,
            label=r"MIMO-GS Top-$K$  $\bar{R}_{\rm eff}(K)$")

    # small star at the actual optimum K* (sweet spot, just above Top-1)
    ax.plot([K_star], [R_star], "*", color="tab:green", ms=16, zorder=5,
            label=rf"Optimum $K^\star={K_star}$ ({R_star:.3f})")

    # headline point 1: Top-1 (predict only) -- 1 probe, sits high
    ax.plot([1], [R_top1], "o", color="tab:blue", ms=12, mec="k", mew=1.3, zorder=6)
    ax.annotate(f"Top-1 (predict only)\n{R_top1:.3f} bits/s/Hz, 1 probe",
                (1, R_top1), textcoords="offset points", xytext=(12, -6),
                fontsize=10, fontweight="bold", color="tab:blue",
                va="top", ha="left")

    # headline point 2: Exhaustive sweep -- 64 probes, sits low
    if feasible[NrNt - 1]:
        ax.plot([NrNt], [R_exh], "s", color="tab:red", ms=12, mec="k", mew=1.3, zorder=6)
        ax.annotate(f"Exhaustive sweep\n{R_exh:.3f} bits/s/Hz, 64 probes",
                    (NrNt, R_exh), textcoords="offset points", xytext=(-10, 8),
                    fontsize=10, fontweight="bold", color="tab:red",
                    va="bottom", ha="right")

    # log2 K-axis with ticks at powers of two
    ax.set_xscale("log", base=2)
    xticks = [1, 2, 4, 8, 16, 32, 64]
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(0.85, 75)

    ax.set_xlabel(r"Probing budget $K$ (beam pairs probed, log$_2$ scale)")
    ax.set_ylabel(r"Net spectral efficiency $\bar{R}_{\rm eff}$ (bits/s/Hz)")
    ax.set_title(rf"Net SE vs probing budget $K$   (SNR = {args.snr:.0f} dB, "
                 rf"$T_c$ = {Tc:.0f}, $L_p$ = {Lp})")
    ax.grid(True, which="both", alpha=0.3)

    # Zoom the y-axis into the top band so the inverted-U is readable: crop from just below
    # the (lowest) exhaustive point up to just above the genie line, instead of starting at 0.
    y_low_ref = R_exh if np.isfinite(R_exh) else float(np.nanmin(Rf))
    span = R_genie_bar - y_low_ref
    margin = 0.05 * span
    ax.set_ylim(y_low_ref - margin, R_genie_bar + margin)
    # With small K bunched near the top and only large K descending, the lower-left is empty.
    ax.legend(loc="lower left", fontsize=10, framealpha=0.92)
    fig.tight_layout()

    stem = f"netse_vs_k_Tc{int(Tc)}"
    pdf, png = C.savefig_pdf_png(fig, out_dir, stem)
    plt.close(fig)

    # CSV
    C.write_csv(
        f"{out_dir}/{stem}.csv",
        ["K", "R_bar_eff", "rho_bar_align"],
        [[K, ("" if not feasible[K-1] else f"{R_bar[K-1]:.6f}"), f"{rho_bar[K-1]:.6f}"]
         for K in K_grid],
    )

    return dict(Tc=Tc, K_star=K_star, R_star=R_star, R_top1=R_top1, R_exh=R_exh,
                pdf=pdf, png=png)


def main():
    args = parse_args()
    out_dir = C.script_out_dir(args.ckpt, "simul_tradeoff_k")
    ctx = C.load_context(ckpt_path=args.ckpt, use_cache=not args.no_cache, device=args.device)

    Lp = args.Lp
    SNR_lin = C.db2lin(args.snr)
    NrNt = ctx.Nr * ctx.Nt  # 64
    K_grid = list(range(1, NrNt + 1))

    # genie ceiling (SNR-fixed, Tc-independent)
    R_genie_bar = float(np.mean(C.R_genie(ctx.g_star, SNR_lin)))

    print(f"[tradeoff_k] SNR={args.snr} dB, Lp={Lp}, Nr={ctx.Nr}, Nt={ctx.Nt}, "
          f"Nr*Nt={NrNt}, Ntest={ctx.ntest}, "
          f"gain_scale={ctx.gain_scale:.4g} (median g_star -> 0 dB)")

    plt = C.setup_matplotlib()

    Tc_list = [float(t) for t in (args.Tc_list if args.Tc_list is not None else [args.Tc])]

    summaries = []
    for Tc in Tc_list:
        R_bar, rho_bar, feasible = sweep_topk(ctx, K_grid, SNR_lin, Lp, Tc)
        s = make_figure(plt, Tc, args, ctx, K_grid, R_bar, rho_bar, feasible,
                        R_genie_bar, Lp, out_dir)
        summaries.append(s)

    # ---- stdout summary ----
    print("\n========== simul_tradeoff_k summary ==========")
    print(f"System: Nr={ctx.Nr}, Nt={ctx.Nt}, Nr*Nt={NrNt}, Ntest={ctx.ntest}")
    print(f"Params: SNR={args.snr:.0f} dB, Lp={Lp}, gain_scale={ctx.gain_scale:.4g} "
          f"(median g_star -> 0 dB)")
    print(f"Genie ceiling (no overhead): R_bar_genie = {R_genie_bar:.4f} bits/s/Hz")
    for s in summaries:
        Tc = s["Tc"]
        R1, Rs, Re = s["R_top1"], s["R_star"], s["R_exh"]
        print(f"\n--- Tc = {Tc:.0f} ---")
        print(f"  Top-1 (K=1,  1 probe ): R_bar_eff = {R1:.4f}  "
              f"({100*R1/R_genie_bar:.1f}% of genie)")
        print(f"  K*    (K={s['K_star']:<2d}, optimum ): R_bar_eff = {Rs:.4f}  "
              f"({100*Rs/R_genie_bar:.1f}% of genie)")
        if np.isfinite(Re):
            print(f"  Exhaustive (K=64, 64 probes): R_bar_eff = {Re:.4f}  "
                  f"({100*Re/R_genie_bar:.1f}% of genie)")
            print(f"  -> Top-1 is x{R1/Re:.2f} above Exhaustive "
                  f"(1 probe vs 64 probes); K* x{Rs/Re:.2f} above Exhaustive")
        print(f"  FIG: {s['pdf']}\n       {s['png']}")
    print(f"\nCSV : " + "  ".join(f"{out_dir}/netse_vs_k_Tc{int(s['Tc'])}.csv" for s in summaries))


if __name__ == "__main__":
    main()
