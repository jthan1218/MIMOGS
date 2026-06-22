"""
mimogs_eval_common.py
=====================

Shared helper module for the MIMO-GS beam-training (beam-management) evaluation.

This module is *imported* by the three simulation scripts:
    - simul_netse_tc.py
    - simul_netse_snr.py
    - simul_alignment.py
and is NOT meant to be run directly.

What it provides
----------------
1. load_model(ckpt_path)
       Rebuild ModelParams/OptimizationParams from the checkpoint, build the Scene,
       restore the GaussianModel (including the dynamic_gain_net MLP), and put the
       model in eval / no_grad mode.

2. render_all_test(...)
       For every held-out test UE location p, render the predicted beamspace magnitude
       map Mhat(p) and read the TRUE map M(p) (raw, NOT normalized). Results are cached
       to .../beam_eval/_cache/rendered_maps.npz and reloaded if present.

3. Gain helpers and a reproducible, dataset-wide gain scaling so the SNR axis is
   interpretable (median(g_star) -> 0 dB).

4. Beam-training primitives:
       candidate_set(Mhat, Delta), topk_set(Mhat, K), select_and_score(C, Mtrue),
       net_se(...), and the R_genie / R_exhaustive references.

5. Convenience batch evaluators (evaluate_delta / evaluate_topk) used by the scripts,
   plus small plotting/IO conventions.

All renders reuse the SAME render kwargs as train.py (rx_shape=(2,2), tx_shape=(4,4),
normalize_beam_weights=False, weight_floor=1e-4, max_active_rx/tx_beams,
renormalize_local_beam_weights) pulled from the restored model_params, so inference
matches training exactly.

Author: MIMO-GS evaluation pipeline.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

# ----------------------------------------------------------------------------
# Repo imports (must be importable from the repo root).
# ----------------------------------------------------------------------------
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.loss import normalize_mag_map


# ============================================================================
# Paths / conventions
# ============================================================================
DEFAULT_CKPT = os.path.join("outputs", "20260609_012844", "model.pth")


def model_dir_from_ckpt(ckpt_path: str) -> str:
    """outputs/<run>/model.pth  ->  outputs/<run>"""
    return os.path.dirname(os.path.abspath(ckpt_path))


def beam_eval_dir(ckpt_path: str) -> str:
    """Root output directory for all evaluation artifacts."""
    return os.path.join(model_dir_from_ckpt(ckpt_path), "beam_eval")


def cache_path(ckpt_path: str) -> str:
    return os.path.join(beam_eval_dir(ckpt_path), "_cache", "rendered_maps.npz")


def script_out_dir(ckpt_path: str, script_name: str) -> str:
    """outputs/<run>/beam_eval/<script_name>/ (created on demand)."""
    d = os.path.join(beam_eval_dir(ckpt_path), script_name)
    os.makedirs(d, exist_ok=True)
    return d


# ============================================================================
# Default system parameters (configurable; STATE them in the report)
# ============================================================================
DEFAULTS = SimpleNamespace(
    Lp=1,            # pilot symbols per probed beam pair
    Tc=256,          # coherence length (symbols)
    SNR_dB=10.0,     # default operating SNR
)


def db2lin(x_db: float | np.ndarray) -> float | np.ndarray:
    return 10.0 ** (np.asarray(x_db, dtype=np.float64) / 10.0)


def lin2db(x_lin: float | np.ndarray) -> float | np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(x_lin, dtype=np.float64), 1e-300))


# ============================================================================
# 1) Model loading / restore
# ============================================================================
def _ns_from_dict(d: dict) -> SimpleNamespace:
    return SimpleNamespace(**dict(d))


@dataclass
class LoadedModel:
    gaussians: GaussianModel
    scene: Scene
    model_params: SimpleNamespace
    opt_params: SimpleNamespace
    tx_pos: torch.Tensor          # (3,) BS / transmitter position, on device
    device: torch.device
    Nr: int
    Nt: int
    beam_rows: int
    beam_cols: int
    # exact render kwargs reused from training
    render_kwargs: dict


def load_model(ckpt_path: str = DEFAULT_CKPT, device: Optional[str] = None) -> LoadedModel:
    """
    Restore a trained MIMO-GS model from a checkpoint and build its Scene.

    Returns a LoadedModel bundle. The GaussianModel is restored INCLUDING the
    dynamic_gain_net MLP via GaussianModel.restore(ckpt['gaussians'], opt_params).
    The model is left in no-grad/eval-friendly state (parameters are not frozen but
    callers must use torch.no_grad()).
    """
    ckpt_path = os.path.abspath(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_params = _ns_from_dict(ckpt["model_params"])
    opt_params = _ns_from_dict(ckpt["opt_params"])

    # Verify the dataset exists *before* doing anything heavy.
    src = getattr(model_params, "source_path", "")
    if not src or not os.path.isdir(src):
        raise FileNotFoundError(
            "Dataset directory referenced by the checkpoint does not exist.\n"
            f"  Expected source_path = {src!r}\n"
            "Place the dataset there (it must contain bs_info.yml, train.mat, test.mat) "
            "or fix model_params['source_path'] in the checkpoint."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    # Build an empty GaussianModel, then restore from capture().
    gaussians = GaussianModel(device=str(device_t))
    gaussians.restore(ckpt["gaussians"], opt_params)

    # Freeze: this is an eval-only pipeline.
    for attr in ("_xyz", "_xyz_tx", "_scaling", "_rotation", "_opacity"):
        p = getattr(gaussians, attr, None)
        if isinstance(p, torch.Tensor):
            p.requires_grad_(False)
    gaussians.dynamic_gain_net.eval()
    for p in gaussians.dynamic_gain_net.parameters():
        p.requires_grad_(False)

    # The Scene only needs model_path + source_path from model_params.
    scene = Scene(model_params, gaussians)

    tx_pos = torch.tensor(scene.bs_position, dtype=torch.float32, device=device_t)

    # Per train.py: rx_shape=(2,2)->Nr=4, tx_shape=(4,4)->Nt=16; map is (beam_rows, beam_cols)
    beam_rows = scene.beam_rows   # 4 (Nr)
    beam_cols = scene.beam_cols   # 16 (Nt)
    Nr = beam_rows
    Nt = beam_cols

    render_kwargs = dict(
        rx_shape=(2, 2),
        tx_shape=(4, 4),
        normalize_beam_weights=False,
        weight_floor=1e-4,
        max_active_rx_beams=getattr(model_params, "max_active_rx_beams", 2),
        max_active_tx_beams=getattr(model_params, "max_active_tx_beams", 2),
        renormalize_local_beam_weights=getattr(model_params, "renormalize_local_beam_weights", True),
    )

    return LoadedModel(
        gaussians=gaussians,
        scene=scene,
        model_params=model_params,
        opt_params=opt_params,
        tx_pos=tx_pos,
        device=device_t,
        Nr=Nr,
        Nt=Nt,
        beam_rows=beam_rows,
        beam_cols=beam_cols,
        render_kwargs=render_kwargs,
    )


# ============================================================================
# 2) Render all held-out test locations (with on-disk cache)
# ============================================================================
@dataclass
class RenderedMaps:
    Mhat: np.ndarray     # [Ntest, Nr, Nt] predicted magnitude maps (raw)
    Mtrue: np.ndarray    # [Ntest, Nr, Nt] true magnitude maps (raw, NOT normalized)
    rx_pos: np.ndarray   # [Ntest, 3] (normalized UE positions, as fed to render)
    Nr: int
    Nt: int


def render_all_test(
    lm: LoadedModel,
    ckpt_path: str = DEFAULT_CKPT,
    use_cache: bool = True,
    verbose: bool = True,
) -> RenderedMaps:
    """
    Render Mhat(p) for every test location and read the raw true map M(p).

    Caches to .../beam_eval/_cache/rendered_maps.npz. Pass use_cache=False to force
    a re-render (the scripts expose this as --no-cache).
    """
    cpath = cache_path(ckpt_path)

    if use_cache and os.path.exists(cpath):
        if verbose:
            print(f"[render_all_test] Loading cached maps from {cpath}")
        data = np.load(cpath)
        return RenderedMaps(
            Mhat=data["Mhat"],
            Mtrue=data["Mtrue"],
            rx_pos=data["rx_pos"],
            Nr=int(data["Nr"]),
            Nt=int(data["Nt"]),
        )

    scene = lm.scene
    gaussians = lm.gaussians
    tx_pos = lm.tx_pos
    device = lm.device
    Nr, Nt = lm.Nr, lm.Nt

    ntest = len(scene.test_set)
    Mhat = np.zeros((ntest, Nr, Nt), dtype=np.float32)
    Mtrue = np.zeros((ntest, Nr, Nt), dtype=np.float32)
    rx_all = np.zeros((ntest, 3), dtype=np.float32)

    if verbose:
        print(f"[render_all_test] Rendering {ntest} test locations ...")

    with torch.no_grad():
        for idx in range(ntest):
            magnitude, rx_pos = scene.test_set[idx]
            rx_pos = rx_pos.to(device)
            magnitude = magnitude.to(device).reshape(scene.beam_rows, scene.beam_cols)

            out = render(
                rx_pos=rx_pos,
                tx_pos=tx_pos,
                pc=gaussians,
                **lm.render_kwargs,
            )
            pred = out["render"]
            # render() returns a real, non-negative magnitude map; take abs() defensively
            # (in case of any tiny negative/complex residue).
            if torch.is_complex(pred):
                pred = pred.abs()
            else:
                pred = pred.abs()

            Mhat[idx] = pred.detach().cpu().numpy()
            Mtrue[idx] = magnitude.detach().cpu().numpy()
            rx_all[idx] = rx_pos.detach().cpu().numpy().reshape(-1)[:3]

            if verbose and (idx + 1) % 500 == 0:
                print(f"    rendered {idx + 1}/{ntest}")

    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    np.savez_compressed(
        cpath,
        Mhat=Mhat,
        Mtrue=Mtrue,
        rx_pos=rx_all,
        Nr=np.int64(Nr),
        Nt=np.int64(Nt),
    )
    if verbose:
        print(f"[render_all_test] Saved cache to {cpath}")

    return RenderedMaps(Mhat=Mhat, Mtrue=Mtrue, rx_pos=rx_all, Nr=Nr, Nt=Nt)


# ============================================================================
# 3) Gain helpers + reproducible dataset-wide gain scaling
# ============================================================================
def gain_from_mag(M: np.ndarray) -> np.ndarray:
    """g[m,n] = M[m,n]**2  (works for a single map or a batch [.., Nr, Nt])."""
    return np.asarray(M, dtype=np.float64) ** 2


def g_star(M: np.ndarray) -> np.ndarray:
    """Best (max) beam-pair gain. Per map: scalar; batched: [Ntest]."""
    g = gain_from_mag(M)
    return g.reshape(*g.shape[:-2], -1).max(axis=-1)


def compute_gain_scale(Mtrue: np.ndarray) -> float:
    """
    Reproducible, dataset-wide gain scaling.

    The dataset magnitude scale is arbitrary, so the absolute gain g=M^2 has no
    physical SNR meaning. We rescale all gains by a single constant so that the
    MEDIAN best-beam gain across the test set maps to 0 dB:

        scale = 1 / median_p( g_star(p) )      (computed on the TRUE maps)

    After scaling, SNR (in dB) is referenced to the typical (median) best-beam gain
    of a test UE, which makes the SNR axis interpretable and identical across all
    three scripts. The scale is computed here, in the common module, from the cached
    true maps -> deterministic and shared.
    """
    gs = g_star(Mtrue)                      # [Ntest], raw
    med = float(np.median(gs))
    if med <= 0.0:
        med = float(np.mean(gs[gs > 0])) if np.any(gs > 0) else 1.0
    return 1.0 / med


# ============================================================================
# Evaluation context bundling rendered maps + scaled gains
# ============================================================================
@dataclass
class EvalContext:
    Mhat: np.ndarray          # [Ntest, Nr, Nt] predicted magnitude (raw)
    Mtrue: np.ndarray         # [Ntest, Nr, Nt] true magnitude (raw)
    g_true: np.ndarray        # [Ntest, Nr, Nt] SCALED true gain (= scale * Mtrue^2)
    g_star: np.ndarray        # [Ntest] SCALED best gain per location
    gain_scale: float         # the single scaling constant
    Nr: int
    Nt: int

    @property
    def ntest(self) -> int:
        return self.Mhat.shape[0]


def build_context(rm: RenderedMaps) -> EvalContext:
    """Bundle rendered maps and apply the shared gain scaling."""
    scale = compute_gain_scale(rm.Mtrue)
    g_true = scale * gain_from_mag(rm.Mtrue)          # [Ntest, Nr, Nt]
    gstar = g_true.reshape(g_true.shape[0], -1).max(axis=-1)  # [Ntest]
    return EvalContext(
        Mhat=rm.Mhat,
        Mtrue=rm.Mtrue,
        g_true=g_true,
        g_star=gstar,
        gain_scale=scale,
        Nr=rm.Nr,
        Nt=rm.Nt,
    )


def load_context(
    ckpt_path: str = DEFAULT_CKPT,
    use_cache: bool = True,
    device: Optional[str] = None,
    verbose: bool = True,
) -> EvalContext:
    """One-shot convenience: load model (if needed), render/cache, build context."""
    cpath = cache_path(ckpt_path)
    if use_cache and os.path.exists(cpath):
        # Avoid loading the heavy model when the cache exists.
        data = np.load(cpath)
        rm = RenderedMaps(
            Mhat=data["Mhat"], Mtrue=data["Mtrue"], rx_pos=data["rx_pos"],
            Nr=int(data["Nr"]), Nt=int(data["Nt"]),
        )
        if verbose:
            print(f"[load_context] Loaded cached maps from {cpath}")
    else:
        lm = load_model(ckpt_path, device=device)
        rm = render_all_test(lm, ckpt_path=ckpt_path, use_cache=use_cache, verbose=verbose)
    return build_context(rm)


# ============================================================================
# 4) Beam-training primitives (single-map, as specified)
# ============================================================================
def candidate_set(Mhat: np.ndarray, Delta: float) -> Tuple[np.ndarray, int]:
    """
    Predicted candidate set for one location:
        C(p) = { (m,n) : Mhat[m,n] >= (1 - Delta) * max(Mhat) },   K(p) = |C(p)| >= 1.

    Returns:
        C    : (K, 2) int array of (m, n) beam-pair indices
        K    : int, number of candidates
    """
    Mhat = np.asarray(Mhat, dtype=np.float64)
    thr = (1.0 - float(Delta)) * Mhat.max()
    mask = Mhat >= thr
    rows, cols = np.nonzero(mask)
    C = np.stack([rows, cols], axis=1)
    return C, int(C.shape[0])


def topk_set(Mhat: np.ndarray, K: int) -> np.ndarray:
    """
    The K beam pairs with the largest predicted magnitude.
    Returns (K, 2) int array of (m, n) indices.
    """
    Mhat = np.asarray(Mhat, dtype=np.float64)
    K = int(min(max(K, 1), Mhat.size))
    flat = Mhat.reshape(-1)
    idx = np.argpartition(flat, -K)[-K:]
    rows, cols = np.unravel_index(idx, Mhat.shape)
    return np.stack([rows, cols], axis=1)


def select_and_score(
    C: np.ndarray,
    Mtrue: np.ndarray,
    g_star_scaled: float,
    gain_scale: float,
) -> Tuple[Tuple[int, int], float, float]:
    """
    Selection grounded in the TRUE channel: among the probed candidate pairs C, pick
    the one with the largest TRUE magnitude (this models physically probing the
    candidates and measuring their received power), then score alignment.

        (mhat, nhat) = argmax_{(m,n) in C} M(p)[m,n]
        ghat         = scale * M(p)[mhat, nhat]^2          (scaled gain)
        rho_align    = ghat / g_star  in (0, 1]

    Returns:
        (mhat, nhat), ghat_scaled, rho_align
    """
    Mtrue = np.asarray(Mtrue, dtype=np.float64)
    vals = Mtrue[C[:, 0], C[:, 1]]
    j = int(np.argmax(vals))
    m, n = int(C[j, 0]), int(C[j, 1])
    ghat = gain_scale * (Mtrue[m, n] ** 2)
    rho = ghat / max(g_star_scaled, 1e-300)
    rho = min(rho, 1.0)
    return (m, n), ghat, rho


def net_se(
    K: int | np.ndarray,
    rho_align: float | np.ndarray,
    g_star_val: float | np.ndarray,
    SNR_lin: float,
    Lp: int = DEFAULTS.Lp,
    Tc: float = DEFAULTS.Tc,
) -> np.ndarray:
    """
    Net spectral efficiency of the beam-training policy:

        R_eff = (1 - K*Lp/Tc) * log2(1 + SNR * rho_align * g_star)

    Only feasible policies (K*Lp < Tc) yield a value; infeasible policies return NaN.
    Vectorized over K / rho_align / g_star_val.
    """
    K = np.asarray(K, dtype=np.float64)
    rho_align = np.asarray(rho_align, dtype=np.float64)
    g_star_val = np.asarray(g_star_val, dtype=np.float64)

    prelog = 1.0 - K * float(Lp) / float(Tc)
    feasible = (K * float(Lp)) < float(Tc)

    rate = np.log2(1.0 + float(SNR_lin) * rho_align * g_star_val)
    R = prelog * rate
    R = np.where(feasible, R, np.nan)
    return R


def R_genie(g_star_val: np.ndarray, SNR_lin: float) -> np.ndarray:
    """Genie no-overhead ceiling: zero probing, perfect alignment (rho=1, prelog=1)."""
    g = np.asarray(g_star_val, dtype=np.float64)
    return np.log2(1.0 + float(SNR_lin) * g)


def R_exhaustive(
    g_star_val: np.ndarray,
    SNR_lin: float,
    Nr: int,
    Nt: int,
    Lp: int = DEFAULTS.Lp,
    Tc: float = DEFAULTS.Tc,
) -> np.ndarray:
    """
    Full beam sweep: probes all Nr*Nt pairs (rho=1) but pays full overhead.
        R = (1 - Nr*Nt*Lp/Tc) * log2(1 + SNR * g_star)
    Infeasible (Nr*Nt*Lp >= Tc) -> NaN.
    """
    g = np.asarray(g_star_val, dtype=np.float64)
    K = Nr * Nt
    prelog = 1.0 - K * float(Lp) / float(Tc)
    feasible = (K * float(Lp)) < float(Tc)
    R = prelog * np.log2(1.0 + float(SNR_lin) * g)
    return np.where(feasible, R, np.nan)


# ============================================================================
# 5) Batch evaluators (used by the scripts; built on the primitives above)
# ============================================================================
def evaluate_delta(
    ctx: EvalContext,
    Delta: float,
    SNR_lin: float,
    Lp: int = DEFAULTS.Lp,
    Tc: float = DEFAULTS.Tc,
) -> Dict[str, np.ndarray]:
    """
    Run the MIMO-GS candidate-set beam-training policy with margin Delta over the whole
    test set, at a given SNR/Lp/Tc.

    Returns per-location arrays and their averages ('_bar' = mean over test locations):
        K        [Ntest]        candidate-set sizes
        rho      [Ntest]        alignment ratios in (0,1]
        R_eff    [Ntest]        net SE (NaN where infeasible)
        K_bar, rho_bar, R_bar_eff   scalars (means over locations; R averaged over feasible)
        feasible [Ntest] bool
    """
    n = ctx.ntest
    K = np.zeros(n, dtype=np.float64)
    rho = np.zeros(n, dtype=np.float64)

    for p in range(n):
        C, k = candidate_set(ctx.Mhat[p], Delta)
        _, _, r = select_and_score(C, ctx.Mtrue[p], ctx.g_star[p], ctx.gain_scale)
        K[p] = k
        rho[p] = r

    R_eff = net_se(K, rho, ctx.g_star, SNR_lin, Lp=Lp, Tc=Tc)
    feasible = (K * float(Lp)) < float(Tc)

    return dict(
        K=K, rho=rho, R_eff=R_eff, feasible=feasible,
        K_bar=float(np.mean(K)),
        rho_bar=float(np.mean(rho)),
        R_bar_eff=float(np.nanmean(R_eff)) if np.any(feasible) else float("nan"),
        frac_feasible=float(np.mean(feasible)),
    )


def evaluate_topk(
    ctx: EvalContext,
    K: int,
    SNR_lin: float,
    Lp: int = DEFAULTS.Lp,
    Tc: float = DEFAULTS.Tc,
) -> Dict[str, np.ndarray]:
    """
    Fixed Top-K probing policy (probe the K best-predicted pairs, select best by true
    channel). Returns per-location rho and net SE plus their averages.
    """
    n = ctx.ntest
    rho = np.zeros(n, dtype=np.float64)
    for p in range(n):
        C = topk_set(ctx.Mhat[p], K)
        _, _, r = select_and_score(C, ctx.Mtrue[p], ctx.g_star[p], ctx.gain_scale)
        rho[p] = r

    Karr = np.full(n, float(K))
    R_eff = net_se(Karr, rho, ctx.g_star, SNR_lin, Lp=Lp, Tc=Tc)
    feasible = (float(K) * float(Lp)) < float(Tc)

    return dict(
        K=K, rho=rho, R_eff=R_eff,
        rho_bar=float(np.mean(rho)),
        R_bar_eff=float(np.nanmean(R_eff)) if feasible else float("nan"),
        feasible=feasible,
    )


def best_delta(
    ctx: EvalContext,
    SNR_lin: float,
    Tc: float,
    Lp: int = DEFAULTS.Lp,
    delta_grid: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Adaptive MIMO-GS: sweep Delta and return the one maximizing R_bar_eff at this
    (SNR, Tc). Returns the winning Delta and its (K_bar, rho_bar, R_bar_eff).
    """
    if delta_grid is None:
        delta_grid = default_delta_grid()

    best = dict(Delta=0.0, R_bar_eff=-np.inf, K_bar=np.nan, rho_bar=np.nan)
    for D in delta_grid:
        res = evaluate_delta(ctx, float(D), SNR_lin, Lp=Lp, Tc=Tc)
        if np.isfinite(res["R_bar_eff"]) and res["R_bar_eff"] > best["R_bar_eff"]:
            best = dict(
                Delta=float(D),
                R_bar_eff=res["R_bar_eff"],
                K_bar=res["K_bar"],
                rho_bar=res["rho_bar"],
            )
    return best


def default_delta_grid(num: int = 41) -> np.ndarray:
    """Delta grid over [0, 1] used consistently by the scripts."""
    return np.linspace(0.0, 1.0, num)


def default_topk_grid() -> List[int]:
    """K = 1, 2, 4, 8, 16, 32, 64 (full sweep at 64 for Nr*Nt=64)."""
    return [1, 2, 4, 8, 16, 32, 64]


# ============================================================================
# 6) Rendering-fidelity metric (NMSE on peak-normalized maps) + interpolation baselines
#    Single source of truth shared by simul_mse.py and simul_mse_density_sweep.py so the
#    metric, normalization and position handling are identical across scripts.
# ============================================================================
def norm_map(x: np.ndarray) -> np.ndarray:
    """
    Per-sample PEAK normalization matching the training loss target:
        normalize_mag_map(x) = x / (amax(x) + eps).
    Applied to a single map (any 2D shape, e.g. 4x16).
    """
    t = torch.from_numpy(np.asarray(x, dtype=np.float64))
    return normalize_mag_map(t).cpu().numpy()


def method_nmse(pred_raw: np.ndarray, true_raw: np.ndarray,
                Mg: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    NMSE for one prediction method on per-sample peak-normalized maps, plus the raw-scale
    optimal-alpha NMSE (shape fidelity, scale-free).

    Args:
        pred_raw : [N, Nr, Nt] raw predicted maps
        true_raw : [N, Nr, Nt] raw GT maps
        Mg       : optional [N, Nr, Nt] pre-normalized GT maps (computed if None)
    Returns dict: nmse_db [N], nmse_raw_db [N], Mh [N,Nr,Nt] (normalized preds),
                  agg_db, agg_raw_db, mean_db, median_db.
    """
    pred_raw = np.asarray(pred_raw, dtype=np.float64)
    true_raw = np.asarray(true_raw, dtype=np.float64)
    N = pred_raw.shape[0]
    if Mg is None:
        Mg = np.stack([norm_map(true_raw[i]) for i in range(N)], axis=0)

    num = np.zeros(N); den = np.zeros(N)
    num_r = np.zeros(N); den_r = np.zeros(N)
    Mh = np.zeros_like(pred_raw)
    for i in range(N):
        mh = norm_map(pred_raw[i]); Mh[i] = mh
        d = mh - Mg[i]
        num[i] = float(np.sum(d * d))
        den[i] = float(np.sum(Mg[i] * Mg[i]))
        a = pred_raw[i]; b = true_raw[i]
        denom = float(np.sum(a * a))
        alpha = (float(np.sum(a * b)) / denom) if denom > 0 else 0.0
        num_r[i] = float(np.sum((alpha * a - b) ** 2))
        den_r[i] = float(np.sum(b * b))

    eps = 1e-12
    nmse_db = 10.0 * np.log10(np.maximum(num / np.maximum(den, eps), eps))
    nmse_raw_db = 10.0 * np.log10(np.maximum(num_r / np.maximum(den_r, eps), eps))
    return dict(
        nmse_db=nmse_db, nmse_raw_db=nmse_raw_db, Mh=Mh,
        agg_db=10.0 * np.log10(np.sum(num) / np.sum(den)),
        agg_raw_db=10.0 * np.log10(np.sum(num_r) / np.sum(den_r)),
        mean_db=float(np.mean(nmse_db)), median_db=float(np.median(nmse_db)),
    )


def knn_predict(train_pos: np.ndarray, train_maps: np.ndarray,
                test_pos: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Distance-weighted kNN baseline over the full Euclidean distance.
        train_pos [Ntrain, D], train_maps [Ntrain, 64], test_pos [Ntest, D]
    Returns predicted maps [Ntest, 4, 16] (distance-weighted average for k>1, nearest for k=1).
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(train_pos)
    dist, idx = tree.query(test_pos, k=k)
    if k == 1:
        pred = train_maps[idx]                                  # [Ntest, 64]
    else:
        dist = np.atleast_2d(dist); idx = np.atleast_2d(idx)
        w = 1.0 / np.maximum(dist, 1e-12)
        w = w / w.sum(axis=1, keepdims=True)
        pred = np.einsum("nk,nkd->nd", w, train_maps[idx])      # [Ntest, 64]
    return pred.reshape(-1, 4, 16)


def linear_predict(train_pos: np.ndarray, train_maps: np.ndarray,
                   test_pos: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Vector-valued LinearNDInterpolator (Delaunay) baseline on the non-degenerate coordinates
    (the dataset is coplanar -> constant dims are dropped so Qhull does not fail). Test points
    outside the convex hull (NaN) fall back to nearest-neighbor.
    Returns (pred [Ntest, 4, 16], n_outside).
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    rng = train_pos.max(axis=0) - train_pos.min(axis=0)
    keep = np.where(rng > 1e-6 * max(rng.max(), 1e-12))[0]
    if keep.size < 2:                       # safety: keep the 2 widest-spread dims
        keep = np.argsort(rng)[::-1][:2]
    tp = train_pos[:, keep]; qp = test_pos[:, keep]

    lin = LinearNDInterpolator(tp, train_maps)      # values [Ntrain, 64]
    vals = lin(qp)                                  # [Ntest, 64], NaN outside hull
    outside = np.isnan(vals).any(axis=1)
    n_outside = int(outside.sum())
    if n_outside > 0:
        near = NearestNDInterpolator(tp, train_maps)
        vals[outside] = near(qp[outside])
    return vals.reshape(-1, 4, 16), n_outside


# ============================================================================
# Small plotting / IO helpers
# ============================================================================
def setup_matplotlib():
    """Configure a headless, paper-quality matplotlib and return the module."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
    })
    return plt


def savefig_pdf_png(fig, out_dir: str, stem: str):
    """Save a figure as both PDF and PNG under out_dir; returns the two paths."""
    pdf = os.path.join(out_dir, f"{stem}.pdf")
    png = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    return pdf, png


def write_csv(path: str, header: List[str], rows: List[list]):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    return path


def common_cli_args(parser):
    """Attach the CLI flags shared by all three simulation scripts."""
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT,
                        help="Path to trained checkpoint (model.pth).")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-render of test maps (ignore the on-disk cache).")
    parser.add_argument("--Lp", type=int, default=DEFAULTS.Lp,
                        help="Pilot symbols per probed beam pair.")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cuda/cpu); default auto.")
    return parser
