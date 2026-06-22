#!/usr/bin/env python
"""
simul_mse.py  --  beamspace-map rendering quality: MIMO-GS vs spatial-interpolation baselines
=============================================================================================

Standalone runnable:
    python simul_mse.py
    python simul_mse.py --knn_k 3 --n_examples 6
    python simul_mse.py --no-cache

What this measures
------------------
How well three methods PREDICT the 4x16 beamspace magnitude map at UNSEEN test locations,
all under the SAME NMSE metric (reported in dB), the SAME normalization, and the SAME test
locations. No PSNR, no SSIM.

  1. MIMO-GS (Proposed): the trained model renders Mhat(p) at each test location
     (render_all_test + .npz cache).
  2. kNN: predict the map at a test location as the (distance-weighted) average of the GT
     maps at the k nearest TRAIN locations (Euclidean in 3D).
  3. Linear: scipy LinearNDInterpolator over the TRAIN locations (Delaunay), vector-valued on
     the 64-dim map; test points outside the convex hull (NaN) fall back to nearest-neighbor.

The baselines are fit ONLY from TRAINING observations (train positions + their TRUE maps);
no rendering. MIMO-GS and the baselines are then scored on the identical test set, so the
comparison is apples-to-apples.

Normalization (matches training supervision)
--------------------------------------------
The training loss (utils/loss.hybrid_magnitude_loss) supervises on the SAMPLE-WISE PEAK-
normalized map (normalize_mag_map(x) = x / (amax(x) + eps)). We compute NMSE on the same
per-sample peak-normalized maps for every method:

    Mh = normalize_mag_map(pred),   Mg = normalize_mag_map(M)
    NMSE_i = ||Mh - Mg||_F^2 / ||Mg||_F^2     -> per-sample dB.

Also reported per method (one secondary number): raw-scale NMSE with the optimal per-sample
scalar alpha_i = <pred_i, M_i> / <pred_i, pred_i> (shape fidelity, scale-free), aggregated dB.

Positions: train and test positions are auto-normalized in the dataset by EACH set's own
scale factor, so they live in different scales. We un-normalize both back to raw meters
(positions * scale_factor) to put train/test distances in ONE shared space. The dataset is
coplanar (constant height), so the LinearNDInterpolator runs on the non-degenerate (x,y)
coordinates while kNN uses the full 3D Euclidean distance.

Reuses mimogs_eval_common.py (load_model -> Scene with train_set/test_set; render_all_test +
.npz cache for Mhat, Mtrue) and utils.loss.normalize_mag_map.

Outputs (under outputs/<run>/beam_eval/simul_mse/):
    mse_cdf.{pdf,png}        -- CDF of per-location NMSE(dB), three methods
    mse_examples.{pdf,png}   -- GT vs MIMO-GS vs kNN vs Linear heatmaps (best/median/worst)
    mse_per_location.csv     -- [idx, nmse_mimogs_db, nmse_knn_db, nmse_linear_db, + raw cols]
"""

import argparse
import numpy as np
import torch

import mimogs_eval_common as C

# Baseline predictors and the NMSE-with-normalize_mag_map metric live in the common module
# (single source of truth). Imported here so simul_mse.py and simul_mse_density_sweep.py share
# identical behavior.
from mimogs_eval_common import norm_map as _norm, method_nmse, knn_predict, linear_predict


METHODS = ["MIMO-GS", "kNN", "Linear"]
METHOD_COLORS = {"MIMO-GS": "tab:blue", "kNN": "tab:orange", "Linear": "tab:green"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Beamspace-map NMSE (dB): MIMO-GS vs kNN/Linear spatial interpolation.")
    C.common_cli_args(p)  # adds --ckpt, --no-cache, --Lp, --device
    p.add_argument("--knn_k", type=int, default=1,
                   help="Number of nearest train neighbors for the kNN baseline (default 1).")
    p.add_argument("--n_examples", type=int, default=6,
                   help="Number of GT-vs-prediction example rows (best/median/worst).")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def figure_cdf(plt, results, out_dir):
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for m in METHODS:
        r = results[m]
        x = np.sort(r["nmse_db"])
        cdf = np.arange(1, len(x) + 1) / len(x)
        col = METHOD_COLORS[m]
        ax.plot(x, cdf, lw=2.2, color=col,
                label=f"{m}  (mean {r['mean_db']:.2f}, median {r['median_db']:.2f} dB)")
        ax.axvline(r["median_db"], color=col, ls="--", lw=1.3, alpha=0.8)
    ax.set_xlabel("Per-location NMSE (dB)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("CDF of beamspace-map prediction NMSE across test locations\n"
                 "(dashed = per-method median)")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9.5)
    fig.tight_layout()
    return C.savefig_pdf_png(fig, out_dir, "mse_cdf"), fig


def select_examples(nmse_db, n):
    """Pick n test samples spanning the MIMO-GS error: a few best/median/worst (honest)."""
    order = np.argsort(nmse_db)
    N = len(order)
    k_worst = int(np.ceil(n / 3.0))
    k_best = int(np.ceil((n - k_worst) / 2.0))
    k_med = n - k_best - k_worst
    sel = [(int(order[r]), "best") for r in range(k_best)]
    mid = N // 2; half = k_med // 2
    for r in range(mid - half, mid - half + k_med):
        sel.append((int(order[int(np.clip(r, 0, N - 1))]), "median"))
    sel += [(int(order[N - k_worst + r]), "worst") for r in range(k_worst)]
    return sel


def figure_examples(plt, sel, results, Mg, out_dir):
    """Rows = samples (best/median/worst); cols = [GT, MIMO-GS, kNN, Linear]."""
    cols = ["GT"] + METHODS
    nrow = len(sel)
    fig, axes = plt.subplots(nrow, len(cols),
                             figsize=(2.1 * len(cols), 2.0 * nrow + 0.6),
                             constrained_layout=True)
    if nrow == 1:
        axes = axes.reshape(1, -1)

    im = None
    for r, (idx, cat) in enumerate(sel):
        maps = {"GT": Mg[idx], "MIMO-GS": results["MIMO-GS"]["Mh"][idx],
                "kNN": results["kNN"]["Mh"][idx], "Linear": results["Linear"]["Mh"][idx]}
        for c, name in enumerate(cols):
            ax = axes[r, c]
            im = ax.imshow(maps[name], vmin=0.0, vmax=1.0, cmap="viridis",
                           aspect="auto", origin="upper", interpolation="nearest")
            ax.set_xticks([0, 5, 10, 15]); ax.set_yticks([0, 1, 2, 3])
            ax.tick_params(labelsize=7.5)
            if r == 0:
                ax.set_title(name, fontsize=11)
            if name != "GT":
                ax.text(0.96, 0.92, f"{results[name]['nmse_db'][idx]:.1f} dB",
                        transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                        color="white",
                        bbox=dict(fc="black", ec="none", alpha=0.55, pad=1.2))
            if c == 0:
                ax.set_ylabel(f"{cat} (idx {idx})\nRx beam index", fontsize=8.5)
            if r == nrow - 1:
                ax.set_xlabel("Tx beam index", fontsize=8.5)

    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.85, pad=0.02)
    cbar.set_label("Normalized magnitude")
    fig.suptitle("Normalized beamspace maps: GT vs predictions "
                 "(rows = best / median / worst by MIMO-GS NMSE)", fontsize=12)
    return C.savefig_pdf_png(fig, out_dir, "mse_examples"), fig


def main():
    args = parse_args()
    out_dir = C.script_out_dir(args.ckpt, "simul_mse")

    # Load the model/scene (needed for train_set) and render/read the test maps (cache).
    lm = C.load_model(args.ckpt, device=args.device)
    rm = C.render_all_test(lm, ckpt_path=args.ckpt, use_cache=not args.no_cache)
    ctx = C.build_context(rm)  # for restating gain scaling
    scene = lm.scene
    Nr, Nt = ctx.Nr, ctx.Nt

    # Raw (un-normalized) positions in a shared space; raw GT maps from the train set.
    tr, te = scene.train_set, scene.test_set
    train_pos = (tr.positions * tr.scale_factor).cpu().numpy().astype(np.float64)
    test_pos = (te.positions * te.scale_factor).cpu().numpy().astype(np.float64)
    train_maps = tr.magnitude.reshape(len(tr), -1).cpu().numpy().astype(np.float64)  # [Ntrain,64]

    print(f"[mse] Nr={Nr}, Nt={Nt}, Nr*Nt={Nr*Nt}, Ntrain={len(tr)}, Ntest={ctx.ntest}, "
          f"knn_k={args.knn_k}, gain_scale={ctx.gain_scale:.4g} (median g_star -> 0 dB; "
          f"not used by this per-sample peak-normalized metric)")

    # Predictions
    true_raw = ctx.Mtrue                                   # [Ntest,4,16]
    knn_raw = knn_predict(train_pos, train_maps, test_pos, args.knn_k)
    lin_raw, n_outside = linear_predict(train_pos, train_maps, test_pos)

    # Shared normalized GT
    Mg = np.stack([_norm(true_raw[i]) for i in range(ctx.ntest)], axis=0)

    results = {
        "MIMO-GS": method_nmse(ctx.Mhat, true_raw, Mg),
        "kNN":     method_nmse(knn_raw,  true_raw, Mg),
        "Linear":  method_nmse(lin_raw,  true_raw, Mg),
    }

    # ---- figures ----
    plt = C.setup_matplotlib()
    (cdf_pdf, _), fcdf = figure_cdf(plt, results, out_dir); plt.close(fcdf)

    n_ex = int(min(max(args.n_examples, 3), ctx.ntest))
    sel = select_examples(results["MIMO-GS"]["nmse_db"], n_ex)
    (ex_pdf, _), fex = figure_examples(plt, sel, results, Mg, out_dir); plt.close(fex)

    # ---- CSV ----
    rd = {m: results[m] for m in METHODS}
    C.write_csv(
        f"{out_dir}/mse_per_location.csv",
        ["idx", "nmse_mimogs_db", "nmse_knn_db", "nmse_linear_db",
         "nmse_mimogs_raw_db", "nmse_knn_raw_db", "nmse_linear_raw_db"],
        [[i,
          f"{rd['MIMO-GS']['nmse_db'][i]:.6f}", f"{rd['kNN']['nmse_db'][i]:.6f}",
          f"{rd['Linear']['nmse_db'][i]:.6f}",
          f"{rd['MIMO-GS']['nmse_raw_db'][i]:.6f}", f"{rd['kNN']['nmse_raw_db'][i]:.6f}",
          f"{rd['Linear']['nmse_raw_db'][i]:.6f}"]
         for i in range(ctx.ntest)],
    )

    # ---- stdout summary ----
    print("\n========== simul_mse summary ==========")
    print(f"System: Nr={Nr}, Nt={Nt}, Nr*Nt={Nr*Nt}, Ntrain={len(tr)}, Ntest={ctx.ntest}, "
          f"gain_scale={ctx.gain_scale:.4g} (median g_star -> 0 dB)")
    print(f"Normalization: per-sample peak (x / amax(x)) -- matches the training loss target.")
    print(f"Linear interpolation: {n_outside}/{ctx.ntest} test points fell OUTSIDE the convex "
          f"hull -> nearest-neighbor fallback.")
    print(f"\n{'method':<9} {'mean dB':>9} {'median dB':>10} {'aggregate dB':>13} {'raw dB':>9}")
    for m in METHODS:
        r = results[m]
        print(f"{m:<9} {r['mean_db']:>9.2f} {r['median_db']:>10.2f} "
              f"{r['agg_db']:>13.2f} {r['agg_raw_db']:>9.2f}")
    base = results["MIMO-GS"]
    print(f"\nMIMO-GS improvement (mean NMSE):")
    for m in ["kNN", "Linear"]:
        d_mean = results[m]["mean_db"] - base["mean_db"]
        d_med = results[m]["median_db"] - base["median_db"]
        print(f"   vs {m:<7}: {d_mean:.2f} dB better (mean), {d_med:.2f} dB better (median)")
    worst = [idx for idx, cat in sel if cat == "worst"]
    print(f"\nWorst MIMO-GS samples (idx): {worst}")
    for idx in worst:
        print(f"   idx={idx}: MIMO-GS={base['nmse_db'][idx]:.1f} dB, "
              f"kNN={results['kNN']['nmse_db'][idx]:.1f} dB, "
              f"Linear={results['Linear']['nmse_db'][idx]:.1f} dB")
    print(f"\nCSV : {out_dir}/mse_per_location.csv")
    print(f"FIG : {cdf_pdf}\n      {ex_pdf}")


if __name__ == "__main__":
    main()
