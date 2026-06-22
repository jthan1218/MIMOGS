#!/usr/bin/env python
"""
simul_alignment.py  --  alignment efficiency vs probing budget, and CDF
=======================================================================

Standalone runnable:
    python simul_alignment.py
    python simul_alignment.py --no-cache
    python simul_alignment.py --cdf_topk 1 2 8

Figure 1 (alignment efficiency vs probing budget):
    rho_bar_align vs probing budget K_bar using fixed Top-K probing (K = 1,2,4,8,...,64).
    Shows how few probes already reach a large fraction of the optimal beam gain; the
    curve saturates toward 1.0. This is an overhead-free view (no T_c, no prelog),
    complementary to the net-SE figures.

Figure 2 (CDF of alignment):
    CDF of rho_align(p) across all test locations for a few Top-K budgets, showing the
    per-location spread (not just the mean).

Outputs (under outputs/<run>/beam_eval/simul_alignment/):
    align_vs_budget.pdf/.png + align_vs_budget.csv
    align_cdf.pdf/.png       + align_cdf_perloc.csv
"""

import argparse
import numpy as np

import mimogs_eval_common as C


def parse_args():
    p = argparse.ArgumentParser(description="Alignment efficiency vs probing budget + CDF.")
    C.common_cli_args(p)
    p.add_argument("--topk", type=int, nargs="+", default=None,
                   help="Top-K budgets for the budget curve (default 1 2 4 8 16 32 64).")
    p.add_argument("--cdf_topk", type=int, nargs="+", default=[1, 2, 4, 8],
                   help="Top-K budgets to draw CDF curves for.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = C.script_out_dir(args.ckpt, "simul_alignment")
    ctx = C.load_context(ckpt_path=args.ckpt, use_cache=not args.no_cache, device=args.device)

    topk_grid = args.topk if args.topk is not None else C.default_topk_grid()
    NrNt = ctx.Nr * ctx.Nt

    # SNR is irrelevant to alignment efficiency (overhead-free view); we only need rho.
    # Use a dummy SNR for the evaluate_topk call; rho does not depend on it.
    snr_dummy = 1.0

    print(f"[alignment] Nr*Nt={NrNt}, Ntest={ctx.ntest}, "
          f"gain_scale={ctx.gain_scale:.4g} (median g_star -> 0 dB)")

    # ========================================================================
    # Figure 1: rho_bar_align vs probing budget K_bar (Top-K)
    # ========================================================================
    K_vals, rho_bar_vals = [], []
    per_loc_rho = {}  # K -> [Ntest] for reuse in CDF
    for K in topk_grid:
        res = C.evaluate_topk(ctx, int(K), snr_dummy, Lp=args.Lp, Tc=C.DEFAULTS.Tc)
        K_vals.append(int(K))
        rho_bar_vals.append(res["rho_bar"])
        per_loc_rho[int(K)] = res["rho"]

    K_vals = np.array(K_vals)
    rho_bar_vals = np.array(rho_bar_vals)

    C.write_csv(
        f"{out_dir}/align_vs_budget.csv",
        ["K", "rho_bar_align"],
        [[k, f"{r:.6f}"] for k, r in zip(K_vals, rho_bar_vals)],
    )

    plt = C.setup_matplotlib()
    fig1, ax1 = plt.subplots(figsize=(7.2, 4.8))
    ax1.plot(K_vals, rho_bar_vals, "-o", color="tab:blue", ms=6, lw=2.0,
             label=r"MIMO-GS Top-$K$ probing")
    ax1.axhline(1.0, color="k", ls="--", lw=1.2, label="Optimal (genie) alignment = 1.0")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(K_vals)
    ax1.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel(r"Probing budget $\bar{K}$ (beam pairs probed)")
    ax1.set_ylabel(r"Alignment efficiency $\bar{\rho}_{\rm align}$")
    ax1.set_title("Alignment efficiency vs probing budget")
    ax1.set_ylim(min(0.5, float(rho_bar_vals.min()) - 0.05), 1.02)
    # annotate each point with its percentage
    for k, r in zip(K_vals, rho_bar_vals):
        ax1.annotate(f"{100*r:.1f}%", (k, r), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=9)
    ax1.legend(loc="lower right")
    fig1.tight_layout()
    pdf1, png1 = C.savefig_pdf_png(fig1, out_dir, "align_vs_budget")

    # ========================================================================
    # Figure 2: CDF of rho_align(p) for a few Top-K budgets
    # ========================================================================
    cdf_ks = [int(k) for k in args.cdf_topk]
    # ensure we have per-loc rho for each requested CDF budget
    for K in cdf_ks:
        if K not in per_loc_rho:
            res = C.evaluate_topk(ctx, K, snr_dummy, Lp=args.Lp, Tc=C.DEFAULTS.Tc)
            per_loc_rho[K] = res["rho"]

    fig2, ax2 = plt.subplots(figsize=(7.2, 4.8))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(cdf_ks)))
    for K, col in zip(cdf_ks, colors):
        rho = np.sort(per_loc_rho[K])
        cdf = np.arange(1, len(rho) + 1) / len(rho)
        ax2.plot(rho, cdf, lw=2.0, color=col,
                 label=rf"$K={K}$ ($\bar\rho={per_loc_rho[K].mean():.3f}$)")
    ax2.set_xlabel(r"Alignment ratio $\rho_{\rm align}(p)$")
    ax2.set_ylabel("Empirical CDF")
    ax2.set_title("CDF of alignment ratio across test locations")
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="upper left")
    fig2.tight_layout()
    pdf2, png2 = C.savefig_pdf_png(fig2, out_dir, "align_cdf")

    # per-location CSV for CDFs
    header = ["loc_idx"] + [f"rho_topk_{K}" for K in cdf_ks]
    rows = []
    for p in range(ctx.ntest):
        rows.append([p] + [f"{per_loc_rho[K][p]:.6f}" for K in cdf_ks])
    C.write_csv(f"{out_dir}/align_cdf_perloc.csv", header, rows)

    # ---- stdout summary ----
    print("\n========== simul_alignment summary ==========")
    print(f"Nr*Nt = {NrNt},  Ntest = {ctx.ntest}")
    for k, r in zip(K_vals, rho_bar_vals):
        print(f"   Top-{k:<2d}:  rho_bar_align = {r:.4f}  ({100*r:.1f}% of optimal)")
    # headline: smallest K reaching 90% / 95%
    for thr in (0.90, 0.95, 0.99):
        idx = np.where(rho_bar_vals >= thr)[0]
        if len(idx):
            print(f"   -> {int(K_vals[idx[0]])} probe(s) reach >= {int(thr*100)}% alignment")
    print(f"CSV : {out_dir}/align_vs_budget.csv\n      {out_dir}/align_cdf_perloc.csv")
    print(f"FIG : {pdf1}\n      {pdf2}")


if __name__ == "__main__":
    main()
