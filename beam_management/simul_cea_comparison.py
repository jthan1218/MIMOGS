#!/usr/bin/env python
"""
simul_cea_comparison.py  --  rendering accuracy in WRF-GS's CEA metric (dB, higher = better)
============================================================================================

Standalone runnable:
    python simul_cea_comparison.py
    python simul_cea_comparison.py --knn_k 1 --no-cache

What this reports
-----------------
The same beamspace-map rendering accuracy as simul_mse.py, but expressed in WRF-GS's
Channel Estimation Accuracy (CEA) convention (dB, HIGHER = better):

    CEA = -10 * log10( ||pred - gt||_F^2 / ||gt||_F^2 ) = -(NMSE in dB).

Three methods on the SAME test set, SAME normalization: MIMO-GS (rendered), kNN(k=1) and
LinearND spatial interpolation of the TRUE train maps.

Two variants per method (which normalization each uses is stated):
  1. CEA_norm      : on the SAMPLE-WISE PEAK-normalized maps (Mh = normalize_mag_map(pred),
                     Mg = normalize_mag_map(gt)). This matches the training supervision.
  2. CEA_raw_alpha : raw maps with the per-sample optimal scalar alpha = <pred,gt>/<pred,pred>
                     (shape fidelity, scale-free). This is the closest analog to WRF-GS's
                     scale-sensitive CEA. NOTE: MIMO-GS's absolute output scale is NOT
                     calibrated (training uses per-sample peak normalization), so an
                     absolute-scale CEA WITHOUT alpha would not be meaningful here -- hence
                     the optimal-alpha variant.

Reporting style mirrors WRF-GS: MEDIAN CEA with 10th / 90th percentiles.

Reuses mimogs_eval_common.py: method_nmse / norm_map (NMSE + normalization), knn_predict /
linear_predict (same coplanar position handling), render_all_test (.npz cache).

IMPORTANT: we do NOT fabricate a WRF-GS column. WRF-GS is not trained on this dataset and its
published numbers are on a different dataset / target / array, so they are not directly
comparable. We only report MIMO-GS and the spatial-interpolation baselines.

Outputs (under outputs/<run>/beam_eval/simul_cea_comparison/):
    cea_cdf.{pdf,png}        -- CDF of per-location CEA_norm (dB), three methods
    cea_box.{pdf,png}        -- box plot of per-location CEA_norm (median/quartiles/10-90)
    cea_per_location.csv     -- [idx, cea_mimogs_db, cea_knn_db, cea_linear_db,
                                 cea_mimogs_raw_db, cea_knn_raw_db, cea_linear_raw_db]
"""

import argparse
import numpy as np

import mimogs_eval_common as C


METHODS = ["MIMO-GS", "kNN", "Linear"]
METHOD_COLORS = {"MIMO-GS": "tab:blue", "kNN": "tab:orange", "Linear": "tab:green"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Rendering accuracy in WRF-GS's CEA metric: MIMO-GS vs kNN/LinearND.")
    C.common_cli_args(p)  # adds --ckpt, --no-cache, --Lp, --device
    p.add_argument("--knn_k", type=int, default=1, help="kNN neighbors (default 1).")
    return p.parse_args()


def pctl(x):
    """Return (median, p10, p90) of an array (dB)."""
    return (float(np.median(x)), float(np.percentile(x, 10)), float(np.percentile(x, 90)))


def figure_cdf(plt, cea_norm, medians, out_dir):
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for m in METHODS:
        x = np.sort(cea_norm[m])
        cdf = np.arange(1, len(x) + 1) / len(x)
        col = METHOD_COLORS[m]
        ax.plot(x, cdf, lw=2.2, color=col, label=f"{m}  (median {medians[m]:.2f} dB)")
        ax.axvline(medians[m], color=col, ls="--", lw=1.3, alpha=0.85)
    ax.set_xlabel("Per-location CEA (dB)   [higher = better]")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("CDF of beamspace-map CEA across test locations\n"
                 r"CEA $= -10\log_{10}(\|pred-gt\|^2/\|gt\|^2)$  (peak-normalized maps; "
                 "dashed = median)")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    return C.savefig_pdf_png(fig, out_dir, "cea_cdf"), fig


def figure_box(plt, cea_norm, out_dir):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    data = [cea_norm[m] for m in METHODS]
    bp = ax.boxplot(data, whis=[10, 90], showfliers=False,
                    widths=0.55, patch_artist=True, medianprops=dict(color="k", lw=2.0))
    ax.set_xticks(range(1, len(METHODS) + 1))
    ax.set_xticklabels(METHODS)
    for patch, m in zip(bp["boxes"], METHODS):
        patch.set_facecolor(METHOD_COLORS[m]); patch.set_alpha(0.45)
    ax.set_ylabel("Per-location CEA (dB)   [higher = better]")
    ax.set_title("CEA distribution per method\n(box = quartiles, whiskers = 10th/90th "
                 "percentile, line = median)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return C.savefig_pdf_png(fig, out_dir, "cea_box"), fig


def main():
    args = parse_args()
    out_dir = C.script_out_dir(args.ckpt, "simul_cea_comparison")

    # Load model/scene (train_set needed) + render/read test maps (cache).
    lm = C.load_model(args.ckpt, device=args.device)
    rm = C.render_all_test(lm, ckpt_path=args.ckpt, use_cache=not args.no_cache)
    ctx = C.build_context(rm)
    scene = lm.scene
    Nr, Nt = ctx.Nr, ctx.Nt

    tr, te = scene.train_set, scene.test_set
    train_pos = (tr.positions * tr.scale_factor).cpu().numpy().astype(np.float64)
    test_pos = (te.positions * te.scale_factor).cpu().numpy().astype(np.float64)
    train_maps = tr.magnitude.reshape(len(tr), -1).cpu().numpy().astype(np.float64)
    true_raw = ctx.Mtrue

    print(f"[cea] Nr={Nr}, Nt={Nt}, Nr*Nt={Nr*Nt}, Ntrain={len(tr)}, Ntest={ctx.ntest}, "
          f"knn_k={args.knn_k}, gain_scale={ctx.gain_scale:.4g} (median g_star -> 0 dB; "
          f"not used by this per-sample peak-normalized metric)")

    # Predictions (reuse common baseline predictors).
    knn_raw = C.knn_predict(train_pos, train_maps, test_pos, k=args.knn_k)
    lin_raw, n_outside = C.linear_predict(train_pos, train_maps, test_pos)

    # Shared normalized GT, then NMSE per method via the common metric.
    Mg = np.stack([C.norm_map(true_raw[i]) for i in range(ctx.ntest)], axis=0)
    nmse = {
        "MIMO-GS": C.method_nmse(ctx.Mhat, true_raw, Mg),
        "kNN":     C.method_nmse(knn_raw,  true_raw, Mg),
        "Linear":  C.method_nmse(lin_raw,  true_raw, Mg),
    }

    # CEA = -(NMSE in dB), per location, for both variants.
    cea_norm = {m: -nmse[m]["nmse_db"] for m in METHODS}        # peak-normalized maps
    cea_raw = {m: -nmse[m]["nmse_raw_db"] for m in METHODS}     # raw maps, optimal alpha

    stat_norm = {m: pctl(cea_norm[m]) for m in METHODS}
    stat_raw = {m: pctl(cea_raw[m]) for m in METHODS}
    medians = {m: stat_norm[m][0] for m in METHODS}

    # ---- figures ----
    plt = C.setup_matplotlib()
    (cdf_pdf, _), fcdf = figure_cdf(plt, cea_norm, medians, out_dir); plt.close(fcdf)
    (box_pdf, _), fbox = figure_box(plt, cea_norm, out_dir); plt.close(fbox)

    # ---- CSV ----
    C.write_csv(
        f"{out_dir}/cea_per_location.csv",
        ["idx", "cea_mimogs_db", "cea_knn_db", "cea_linear_db",
         "cea_mimogs_raw_db", "cea_knn_raw_db", "cea_linear_raw_db"],
        [[i,
          f"{cea_norm['MIMO-GS'][i]:.6f}", f"{cea_norm['kNN'][i]:.6f}",
          f"{cea_norm['Linear'][i]:.6f}",
          f"{cea_raw['MIMO-GS'][i]:.6f}", f"{cea_raw['kNN'][i]:.6f}",
          f"{cea_raw['Linear'][i]:.6f}"]
         for i in range(ctx.ntest)],
    )

    # ---- stdout summary ----
    print("\n========== simul_cea_comparison summary ==========")
    print(f"System: Nr={Nr}, Nt={Nt}, Nr*Nt={Nr*Nt}, Ntrain={len(tr)}, Ntest={ctx.ntest}, "
          f"gain_scale={ctx.gain_scale:.4g} (median g_star -> 0 dB)")
    print(f"Normalization: per-sample peak (x / amax(x)) -- matches the training loss target.")
    print(f"LinearND: {n_outside}/{ctx.ntest} test points outside convex hull -> nearest fallback.")
    print(f"CEA = -10*log10(||pred-gt||^2/||gt||^2) = -(NMSE dB); HIGHER is better.\n")
    print(f"{'method':<9} | {'CEA_norm  median (p10/p90)':<30} | {'CEA_raw_alpha  median (p10/p90)'}")
    for m in METHODS:
        mn, l10, l90 = stat_norm[m]
        rm_, r10, r90 = stat_raw[m]
        print(f"{m:<9} | {mn:6.2f} dB ({l10:.2f}/{l90:.2f}){'':<6} | "
              f"{rm_:6.2f} dB ({r10:.2f}/{r90:.2f})")

    print("\nCaveats:")
    print("  - This expresses MIMO-GS accuracy in WRF-GS's CEA metric/convention, so the "
          "numbers are comparable in LANGUAGE only.")
    print("  - A true head-to-head needs WRF-GS TRAINED on THIS dataset and adapted to the "
          "dual-sided MIMO beamspace target; published WRF-GS numbers (different dataset/"
          "target/array) are NOT directly comparable. No WRF-GS column is fabricated.")
    print("  - Scale caveat: MIMO-GS output scale is uncalibrated (per-sample peak norm), so "
          "CEA_raw_alpha uses the optimal per-sample scalar; absolute-scale CEA is not "
          "meaningful here.")
    print(f"\nCSV : {out_dir}/cea_per_location.csv")
    print(f"FIG : {cdf_pdf}\n      {box_pdf}")


if __name__ == "__main__":
    main()
