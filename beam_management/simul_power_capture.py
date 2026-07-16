"""
simul_power_capture.py
======================

Power-capture evaluation for rendered long-term beamspace statistics.

Run from the MIMO-GS repo root:
    python simul_power_capture.py

Outputs:
    outputs/<run>/beam_eval/simul_power_capture/power_capture.png
    outputs/<run>/beam_eval/simul_power_capture/power_capture.pdf
    outputs/<run>/beam_eval/simul_power_capture/power_capture.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

# Make the script robust when executed from outside the repo root.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mimogs_eval_common import (  # noqa: E402
    DEFAULT_CKPT,
    common_cli_args,
    load_context,
    load_model,
    script_out_dir,
    setup_matplotlib,
    savefig_pdf_png,
    write_csv,
)


EPS = 1e-12


def resolve_ckpt(path: str) -> str:
    """Resolve a checkpoint path relative to this script location."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def sanitize_score(x: np.ndarray) -> np.ndarray:
    """Ranking score with invalid entries pushed to the bottom."""
    x = np.asarray(x, dtype=np.float64)
    return np.nan_to_num(x, nan=-np.inf, posinf=np.finfo(np.float64).max, neginf=-np.inf)


def magnitude_to_power(mag: np.ndarray) -> np.ndarray:
    """Beamspace magnitude map -> beamspace power map."""
    mag = np.asarray(mag, dtype=np.float64)
    mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(mag, 0.0) ** 2


def power_capture_curve(
    score_maps: np.ndarray,
    true_power_maps: np.ndarray,
    L_values: Iterable[int],
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Average true-power fraction captured by the top-L beam pairs selected using score_maps.

    Args:
        score_maps      : [N, Nr, Nt], maps used only for beam-pair ranking.
        true_power_maps : [N, Nr, Nt], ground-truth beamspace power maps.
        L_values        : selected beam-pair budgets.
        valid_mask      : optional [N] mask for nonzero-power test samples.
    """
    score = sanitize_score(score_maps)
    power = np.asarray(true_power_maps, dtype=np.float64)

    if score.shape != power.shape:
        raise ValueError(f"score_maps and true_power_maps shape mismatch: {score.shape} vs {power.shape}")

    n_samples = power.shape[0]
    flat_score = score.reshape(n_samples, -1)
    flat_power = power.reshape(n_samples, -1)

    total_power = flat_power.sum(axis=1)
    if valid_mask is None:
        valid_mask = total_power > EPS
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if not np.any(valid_mask):
        raise ValueError("No nonzero-power test samples were found.")

    flat_score = flat_score[valid_mask]
    flat_power = flat_power[valid_mask]
    total_power = total_power[valid_mask]

    # Descending sort by score; true power is accumulated in the selected order.
    order = np.argsort(flat_score, axis=1)[:, ::-1]
    selected_power = np.take_along_axis(flat_power, order, axis=1)
    cumulative_power = np.cumsum(selected_power, axis=1)

    n_beams = flat_power.shape[1]
    curve = []
    for L in L_values:
        L = int(np.clip(L, 1, n_beams))
        ratio = cumulative_power[:, L - 1] / np.maximum(total_power, EPS)
        curve.append(float(np.mean(ratio)))

    return np.asarray(curve, dtype=np.float64)


def nearest_indices(train_pos: np.ndarray, test_pos: np.ndarray) -> np.ndarray:
    """Nearest-neighbor indices with scipy KD-tree and a NumPy fallback."""
    train_pos = np.asarray(train_pos, dtype=np.float64)
    test_pos = np.asarray(test_pos, dtype=np.float64)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(train_pos)
        _, idx = tree.query(test_pos, k=1)
        return np.asarray(idx, dtype=np.int64)
    except Exception as exc:  # pragma: no cover - fallback for minimal environments
        print(f"[nearest] scipy KD-tree unavailable; using chunked NumPy search ({exc}).")
        n_train = train_pos.shape[0]
        n_test = test_pos.shape[0]
        max_pairs = 10_000_000
        chunk = max(1, min(n_test, max_pairs // max(n_train, 1)))
        idx = np.empty(n_test, dtype=np.int64)
        for s in range(0, n_test, chunk):
            e = min(s + chunk, n_test)
            d2 = np.sum((test_pos[s:e, None, :] - train_pos[None, :, :]) ** 2, axis=-1)
            idx[s:e] = np.argmin(d2, axis=1)
        return idx


def nearest_observation_scores(
    ckpt_path: str,
    n_expected: int,
    Nr: int,
    Nt: int,
    device: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    For each test location, return the beamspace map of the nearest observed train location.

    This baseline requires the checkpoint and dataset because rendered-map cache does not
    store train-set maps. If loading fails, the caller can skip the baseline.
    """
    try:
        lm = load_model(ckpt_path, device=device)
        scene = lm.scene
        train_pos = scene.train_set.positions.detach().cpu().numpy()
        test_pos = scene.test_set.positions.detach().cpu().numpy()
        train_maps = scene.train_set.magnitude.detach().cpu().numpy().reshape(-1, Nr, Nt)

        if test_pos.shape[0] != n_expected:
            print(
                f"[nearest] test-set length mismatch: scene={test_pos.shape[0]}, "
                f"rendered={n_expected}. Skipping nearest baseline."
            )
            return None

        idx = nearest_indices(train_pos, test_pos)
        return train_maps[idx]
    except Exception as exc:
        print(f"[nearest] Skipping nearest-observation baseline: {exc}")
        return None


def evaluate_power_capture(
    ckpt_path: str,
    use_cache: bool,
    include_nearest: bool,
    device: Optional[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], int, int, int]:
    """Compute power-capture curves for oracle, rendered statistic, nearest, and random."""
    ctx = load_context(ckpt_path=ckpt_path, use_cache=use_cache, device=device, verbose=True)

    true_power = magnitude_to_power(ctx.Mtrue)
    rendered_score = sanitize_score(ctx.Mhat)
    oracle_score = true_power

    n_samples, Nr, Nt = ctx.Mtrue.shape
    n_beams = Nr * Nt
    L_values = np.arange(1, n_beams + 1, dtype=np.int64)

    total_power = true_power.reshape(n_samples, -1).sum(axis=1)
    valid = total_power > EPS
    n_valid = int(np.sum(valid))

    curves: Dict[str, np.ndarray] = {
        "Oracle": power_capture_curve(oracle_score, true_power, L_values, valid),
        "MIMO-GS": power_capture_curve(rendered_score, true_power, L_values, valid),
        "Random": L_values.astype(np.float64) / float(n_beams),
    }

    if include_nearest:
        nearest_score = nearest_observation_scores(
            ckpt_path=ckpt_path,
            n_expected=n_samples,
            Nr=Nr,
            Nt=Nt,
            device=device,
        )
        if nearest_score is not None:
            curves["Nearest"] = power_capture_curve(nearest_score, true_power, L_values, valid)

    return L_values, curves, Nr, Nt, n_valid


def plot_power_capture(
    L_values: np.ndarray,
    curves: Dict[str, np.ndarray],
    out_dir: str,
    stem: str = "power_capture",
) -> Tuple[str, str, str]:
    """Plot and save the power-capture curve."""
    plt = setup_matplotlib()

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    order = ["Oracle", "MIMO-GS", "Nearest", "Random"]
    markevery = max(1, len(L_values) // 8)

    for name in order:
        if name not in curves:
            continue
        if name == "Random":
            ax.plot(L_values, curves[name], linestyle="--", label=name)
        else:
            ax.plot(L_values, curves[name], marker="o", markevery=markevery, linewidth=2.0, label=name)

    ax.set_title("Beamspace Power Capture")
    ax.set_xlabel("Selected beam pairs")
    ax.set_ylabel("Captured power ratio")
    ax.set_xlim(float(L_values[0]), float(L_values[-1]))
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=True)
    fig.tight_layout()

    pdf_path, png_path = savefig_pdf_png(fig, out_dir, stem)
    plt.close(fig)

    rows = []
    names = [name for name in order if name in curves]
    for i, L in enumerate(L_values):
        rows.append([int(L), float(L / L_values[-1])] + [float(curves[name][i]) for name in names])
    csv_path = write_csv(
        os.path.join(out_dir, f"{stem}.csv"),
        ["L", "budget_fraction"] + names,
        rows,
    )
    return pdf_path, png_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate captured beamspace power ratio versus selected beam-pair budget."
    )
    common_cli_args(parser)
    parser.add_argument(
        "--no-nearest",
        action="store_true",
        help="Skip the nearest-observation baseline.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ckpt_path = resolve_ckpt(args.ckpt if args.ckpt else DEFAULT_CKPT)
    out_dir = script_out_dir(ckpt_path, "simul_power_capture")

    L_values, curves, Nr, Nt, n_valid = evaluate_power_capture(
        ckpt_path=ckpt_path,
        use_cache=not args.no_cache,
        include_nearest=not args.no_nearest,
        device=args.device,
    )
    pdf_path, png_path, csv_path = plot_power_capture(L_values, curves, out_dir)

    print("[simul_power_capture] Done")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  beamspace  : {Nr} x {Nt} ({Nr * Nt} beam pairs)")
    print(f"  valid test : {n_valid}")
    print(f"  figure png : {png_path}")
    print(f"  figure pdf : {pdf_path}")
    print(f"  csv        : {csv_path}")


if __name__ == "__main__":
    main()
