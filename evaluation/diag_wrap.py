"""Beam-coordinate wrap-around exposure diagnostic (read-only).

The renderer measures the distance between a projected primitive centre ``u``
and a DFT beam centre ``b`` as the raw difference ``b - u``.  The array response
phase ``exp(j*pi*k*(u-b))`` is periodic in ``(u-b)`` with period 2, so the
correct distance is the one wrapped into ``[-1,1)``:

    d = (b - u + 1) mod 2 - 1

This script does not change the renderer.  It reuses the renderer's own
projection and top-k selection code to measure how often the two conventions
disagree on the ASU test split, and how much of that disagreement lands on
primitives that actually carry gain.

Run with no arguments:

    python diag_wrap.py
"""
from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The renderer's own code -- imported, never re-implemented.
from gaussian_renderer.fast_renderer import (
    _build_beam_uv_grid,
    _projected_angular_covariance_batched,
    render_fast,
)
from mimogs_rasterizer.reference import topk_gaussian_beam_weights
from eval_render import (
    build_scene_and_model,
    gain_net_hidden_dim,
    gain_net_width,
    restore_config,
)


RUN_DIR = "outputs/20260805_051724"
CHECKPOINT = os.path.join(RUN_DIR, "model.pth")
OUTPUT_DIR = os.path.join("analysis", "wrap_diagnostic")

LOCATION_CHUNK = 64          # test locations projected at once
DOMINANT_FRACTION = 0.10     # "top 10% gain" definition for Step 2.4
COVARIANCE_FLOOR = 1e-4      # matches train.py / eval_render.py
WEIGHT_FLOOR = 1e-4          # matches train.py / eval_render.py

# Verbatim Step-1 audit, reproduced in the README so the numbers below are
# never read apart from the code claim they quantify.
AUDIT_TEXT = """\
STEP 1 -- CODE AUDIT (verbatim)
================================================================

VERDICT: the renderer does NOT wrap the beam-coordinate difference modulo 2
anywhere.  All three code paths form the deviation as a raw difference.

Beam centres are built from the unshifted DFT grid:
  gaussian_renderer/__init__.py:32
      return 2.0 * torch.fft.fftfreq(num_elem, d=1.0, device=device).to(dtype)
  gaussian_renderer/fast_renderer.py:27
      return 2.0 * torch.fft.fftfreq(num_elem, d=1.0, device=device).to(dtype)

(a) QUADRATIC-FORM DEVIATION (the Gaussian weight exponent)

  gaussian_renderer/__init__.py:229-236   (dense reference renderer)
      dx = beam_centers_uv[:, 0].unsqueeze(0) - uv_mean[:, 0].unsqueeze(1)
      dy = beam_centers_uv[:, 1].unsqueeze(0) - uv_mean[:, 1].unsqueeze(1)
      ...
      mahal = inv00 * dx * dx + 2.0 * inv01 * dx * dy + inv11 * dy * dy

  mimogs_rasterizer/reference.py:56-63    (top-k reference rasterizer)
      dx = beam_centers_uv[:, 0] - uv_mean[..., 0, None]
      dy = beam_centers_uv[:, 1] - uv_mean[..., 1, None]
      ...
      mahal = p00 * dx.square() + 2.0 * p01 * dx * dy + p11 * dy.square()

  mimogs_rasterizer/csrc/rasterizer_cuda.cu:95-97   (fused CUDA, tx forward)
      const float dx = centers[2 * b + 0] - ux;
      const float dy = centers[2 * b + 1] - uy;
      const float mahal = p00 * dx * dx + 2.0f * p01 * dx * dy + p11 * dy * dy;

  mimogs_rasterizer/csrc/rasterizer_cuda.cu:143-145 (fused CUDA, rx forward)
      identical expression.

  mimogs_rasterizer/csrc/rasterizer_cuda.cu:318-319 and 400-401 (backward)
      identical expression, so the gradient is consistent with the forward.

(b) CANDIDATE (TOP-K) BEAM SELECTION

  Selection is not separate code: it runs on the logits produced by exactly the
  deviation above, so it inherits the same convention.

  mimogs_rasterizer/reference.py:66-70
      logits = -0.5 * mahal
      top_logits, top_indices = torch.topk(logits, k=k_eff, dim=-1, ...)

  mimogs_rasterizer/csrc/rasterizer_cuda.cu:98 and 146
      insert_topk(-0.5f * mahal, b, top_values, top_indices, k);

  gaussian_renderer/__init__.py:281 (dense path) tops-k the exponentiated
  weights, which is order-equivalent to topping-k the same raw logits.

A repository-wide search for fmod / remainder / "% 2" / wrap / modulo over
gaussian_renderer/, mimogs_rasterizer/, scene/ and utils/ returns no
wrap-related code.

The training path is train.py:18 -> render_fast with use_cuda_rasterizer=1
(run_args.txt), and the CUDA extension is built and active on this machine, so
the fused kernel above is what actually ran.

Grid geometry: 2*fftfreq(N) places bins at exactly -1.0 and at 1 - 2/N, with
spacing 2/N.  Wrap can therefore only change the nearest beam for coordinates
above 1 - 1/N (the bottom edge is exactly on a bin, so it has no gap).  For
this dataset the rx array is 4x4 (16 beams) and the tx array is 8x8 (64 beams),
giving wrap-sensitive bands u > 0.75 and u > 0.875 respectively.
"""


# ----------------------------------------------------------------------
# Wrapping helpers
# ----------------------------------------------------------------------
def wrap_to_unit(delta: torch.Tensor) -> torch.Tensor:
    """Wrap a beam-coordinate difference into ``[-1, 1)``."""
    return torch.remainder(delta + 1.0, 2.0) - 1.0


def beam_deltas(
    uv: torch.Tensor, centers: torch.Tensor, wrap: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(dx, dy)`` between every projected mean and every beam centre."""
    dx = centers[:, 0] - uv[..., 0, None]
    dy = centers[:, 1] - uv[..., 1, None]
    if wrap:
        dx = wrap_to_unit(dx)
        dy = wrap_to_unit(dy)
    return dx, dy


def mahalanobis_logits(
    uv: torch.Tensor, precision: torch.Tensor, centers: torch.Tensor, wrap: bool
) -> torch.Tensor:
    """The renderer's own selection logits, optionally with wrapped deltas."""
    dx, dy = beam_deltas(uv, centers, wrap)
    p00 = precision[..., 0, None]
    p01 = precision[..., 1, None]
    p11 = precision[..., 2, None]
    return -0.5 * (p00 * dx.square() + 2.0 * p01 * dx * dy + p11 * dy.square())


def euclidean_nearest(
    uv: torch.Tensor, centers: torch.Tensor, wrap: bool
) -> torch.Tensor:
    """Nearest beam under the plain grid metric (covariance ignored)."""
    dx, dy = beam_deltas(uv, centers, wrap)
    return (dx.square() + dy.square()).argmin(dim=-1)


# ----------------------------------------------------------------------
# Consistency check against the renderer's own output
# ----------------------------------------------------------------------
def verify_against_renderer(
    gaussians,
    scene,
    model_params: Namespace,
    rx_positions: torch.Tensor,
    tx_pos: torch.Tensor,
    device: torch.device,
) -> List[str]:
    """Confirm the recomputed projection reproduces the renderer exactly.

    Two checks: (1) the projected uv/precision fed to the top-k reference
    reproduce the reference rasterizer's own indices bit-for-bit, and (2) the
    fused CUDA renderer that training used agrees with the reference path, so
    the reference is a faithful stand-in for the diagnostic.
    """
    lines: List[str] = []
    sample = rx_positions[:8].to(device)

    means_rx = gaussians.get_xyz
    covariances = gaussians.get_covariance()
    rx_uv, rx_precision = _projected_angular_covariance_batched(
        means_rx, covariances, sample, COVARIANCE_FLOOR
    )
    rx_uv = -rx_uv

    rx_centers = _build_beam_uv_grid(
        scene.rx_shape[0], scene.rx_shape[1], device=device, dtype=means_rx.dtype
    )
    k_rx = int(getattr(model_params, "max_active_rx_beams", 2))

    # (1) our logits vs the renderer's own top-k routine
    reference_weights, reference_indices = topk_gaussian_beam_weights(
        rx_uv, rx_precision, rx_centers, k_rx, WEIGHT_FLOOR
    )
    ours = mahalanobis_logits(rx_uv, rx_precision, rx_centers, wrap=False)
    ours_top1 = ours.argmax(dim=-1)
    reference_top1 = reference_indices.gather(
        -1, reference_weights.argmax(dim=-1, keepdim=True)
    ).squeeze(-1)
    match = float((ours_top1 == reference_top1).float().mean().item())
    lines.append(
        f"recomputed top-1 vs mimogs_rasterizer.reference top-1 : {match:.6f} agreement"
    )

    # (2) fused CUDA (what training ran) vs the reference path
    with torch.no_grad():
        cuda_out = render_fast(
            rx_pos=sample,
            tx_pos=tx_pos,
            pc=gaussians,
            rx_shape=scene.rx_shape,
            tx_shape=scene.tx_shape,
            covariance_floor=COVARIANCE_FLOOR,
            weight_floor=WEIGHT_FLOOR,
            max_active_rx_beams=k_rx,
            max_active_tx_beams=int(getattr(model_params, "max_active_tx_beams", 2)),
            use_cuda_rasterizer=True,
        )["render"]
        ref_out = render_fast(
            rx_pos=sample,
            tx_pos=tx_pos,
            pc=gaussians,
            rx_shape=scene.rx_shape,
            tx_shape=scene.tx_shape,
            covariance_floor=COVARIANCE_FLOOR,
            weight_floor=WEIGHT_FLOOR,
            max_active_rx_beams=k_rx,
            max_active_tx_beams=int(getattr(model_params, "max_active_tx_beams", 2)),
            use_cuda_rasterizer=False,
        )["render"]
    denom = ref_out.abs().max().clamp_min(1e-20)
    rel = float(((cuda_out - ref_out).abs().max() / denom).item())
    lines.append(f"fused CUDA vs reference renderer            : max rel diff {rel:.3e}")
    return lines


# ----------------------------------------------------------------------
# Main diagnostic
# ----------------------------------------------------------------------
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[diag_wrap] checkpoint : {CHECKPOINT}")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model_params, opt_params = restore_config(RUN_DIR, checkpoint)
    hidden_dim = gain_net_hidden_dim(checkpoint)
    with gain_net_width(hidden_dim):
        scene, gaussians = build_scene_and_model(
            model_params, opt_params, checkpoint, device
        )

    rx_shape, tx_shape = scene.rx_shape, scene.tx_shape
    k_rx = int(getattr(model_params, "max_active_rx_beams", 2))
    k_tx = int(getattr(model_params, "max_active_tx_beams", 2))
    print(f"[diag_wrap] rx array {rx_shape} ({scene.beam_rows} beams), k_rx={k_rx}")
    print(f"[diag_wrap] tx array {tx_shape} ({scene.beam_cols} beams), k_tx={k_tx}")

    test_positions = scene.test_set.positions.to(device)
    num_locations = int(test_positions.shape[0])
    tx_pos = torch.as_tensor(scene.bs_position, dtype=torch.float32, device=device)

    means_rx = gaussians.get_xyz.detach()
    num_gaussians = int(means_rx.shape[0])
    print(f"[diag_wrap] {num_locations} test locations x {num_gaussians} primitives")

    verification = verify_against_renderer(
        gaussians, scene, model_params, test_positions, tx_pos, device
    )
    for line in verification:
        print(f"[diag_wrap] {line}")

    rx_centers = _build_beam_uv_grid(
        rx_shape[0], rx_shape[1], device=device, dtype=means_rx.dtype
    )
    tx_centers = _build_beam_uv_grid(
        tx_shape[0], tx_shape[1], device=device, dtype=means_rx.dtype
    )

    covariances = gaussians.get_covariance().detach()

    # ------------------------------------------------------------------
    # Tx side is location-independent (fixed BS + fixed tx anchors).
    # ------------------------------------------------------------------
    with torch.no_grad():
        tx_uv_b, tx_precision_b = _projected_angular_covariance_batched(
            gaussians.get_xyz_tx.detach(),
            gaussians.get_covariance_tx().detach(),
            tx_pos.view(1, 3),
            COVARIANCE_FLOOR,
        )
    tx_uv = tx_uv_b.squeeze(0)
    tx_precision = tx_precision_b.squeeze(0)

    tx_raw_logits = mahalanobis_logits(tx_uv, tx_precision, tx_centers, wrap=False)
    tx_wrap_logits = mahalanobis_logits(tx_uv, tx_precision, tx_centers, wrap=True)
    tx_top1_disagree = (
        tx_raw_logits.argmax(dim=-1) != tx_wrap_logits.argmax(dim=-1)
    )  # (N,)
    tx_topk_disagree = _topk_set_disagreement(tx_raw_logits, tx_wrap_logits, k_tx)
    tx_geo_disagree = euclidean_nearest(
        tx_uv, tx_centers, False
    ) != euclidean_nearest(tx_uv, tx_centers, True)

    tx_edge_band = 1.0 - 1.0 / float(tx_shape[0])
    tx_edge_band_v = 1.0 - 1.0 / float(tx_shape[1])
    tx_in_band = (tx_uv[:, 0] > tx_edge_band) | (tx_uv[:, 1] > tx_edge_band_v)
    tx_outside = (tx_uv.abs() > 1.0).any(dim=-1)

    # ------------------------------------------------------------------
    # Rx side, chunked over test locations.
    # ------------------------------------------------------------------
    rx_edge_band = 1.0 - 1.0 / float(rx_shape[0])
    rx_edge_band_v = 1.0 - 1.0 / float(rx_shape[1])

    accumulator = _Accumulator(num_locations)

    with torch.no_grad():
        for start in range(0, num_locations, LOCATION_CHUNK):
            stop = min(start + LOCATION_CHUNK, num_locations)
            positions = test_positions[start:stop]
            chunk = stop - start

            rx_uv, rx_precision = _projected_angular_covariance_batched(
                means_rx, covariances, positions, COVARIANCE_FLOOR
            )
            rx_uv = -rx_uv

            gain = gaussians.get_dynamic_gain_weight_batched(positions).abs()  # (B,N)
            threshold = torch.quantile(
                gain.float(), 1.0 - DOMINANT_FRACTION, dim=1, keepdim=True
            )
            dominant = gain >= threshold  # (B,N)

            raw_logits = mahalanobis_logits(rx_uv, rx_precision, rx_centers, False)
            wrap_logits = mahalanobis_logits(rx_uv, rx_precision, rx_centers, True)

            rx_top1 = raw_logits.argmax(dim=-1) != wrap_logits.argmax(dim=-1)
            rx_topk = _topk_set_disagreement(raw_logits, wrap_logits, k_rx)
            rx_geo = euclidean_nearest(rx_uv, rx_centers, False) != euclidean_nearest(
                rx_uv, rx_centers, True
            )
            rx_band = (rx_uv[..., 0] > rx_edge_band) | (rx_uv[..., 1] > rx_edge_band_v)
            rx_out = (rx_uv.abs() > 1.0).any(dim=-1)

            accumulator.add_rx(
                start,
                chunk,
                rx_top1,
                rx_topk,
                rx_geo,
                rx_band,
                rx_out,
                dominant,
            )
            accumulator.add_tx(
                start,
                chunk,
                tx_top1_disagree,
                tx_topk_disagree,
                tx_geo_disagree,
                tx_in_band,
                tx_outside,
                dominant,
            )

            if (start // LOCATION_CHUNK) % 10 == 0:
                print(f"  {stop} / {num_locations} locations")

    stats = accumulator.finalize(num_gaussians)
    stats["tx_top1_disagree_primitives"] = float(tx_top1_disagree.float().mean().item())
    stats["tx_topk_disagree_primitives"] = float(tx_topk_disagree.float().mean().item())

    _write_stats_csv(stats, scene, model_params, num_locations, num_gaussians)
    _write_scatter(accumulator, test_positions, scene)
    _write_readme(stats, verification, scene, num_locations, num_gaussians)
    _print_summary(stats, accumulator, test_positions)


def _topk_set_disagreement(
    raw_logits: torch.Tensor, wrap_logits: torch.Tensor, k: int
) -> torch.Tensor:
    """True where the retained top-k beam SET differs between conventions."""
    num_beams = raw_logits.shape[-1]
    k_eff = min(int(k), num_beams)
    raw_idx = raw_logits.topk(k_eff, dim=-1).indices
    wrap_idx = wrap_logits.topk(k_eff, dim=-1).indices

    raw_mask = torch.zeros_like(raw_logits, dtype=torch.bool)
    wrap_mask = torch.zeros_like(wrap_logits, dtype=torch.bool)
    raw_mask.scatter_(-1, raw_idx, True)
    wrap_mask.scatter_(-1, wrap_idx, True)
    return (raw_mask != wrap_mask).any(dim=-1)


class _Accumulator:
    """Streaming counters over (location, primitive) pairs."""

    def __init__(self, num_locations: int) -> None:
        self.num_locations = num_locations
        self.total_pairs = 0
        self.dominant_pairs = 0
        self.counts: Dict[str, int] = {}
        # Per-location dominant-primitive disagreement counts, for the scatter.
        self.per_location_rx = np.zeros(num_locations, dtype=np.int64)
        self.per_location_tx = np.zeros(num_locations, dtype=np.int64)
        # The dominance mask is ``gain >= quantile(gain, 0.9)``; ties at the
        # threshold make the retained set larger than 10% at some locations, so
        # the denominator has to be recorded per location rather than assumed.
        self.per_location_dominant = np.zeros(num_locations, dtype=np.int64)

    def _bump(self, key: str, value: int) -> None:
        self.counts[key] = self.counts.get(key, 0) + value

    def _add_side(
        self,
        side: str,
        start: int,
        chunk: int,
        top1: torch.Tensor,
        topk: torch.Tensor,
        geo: torch.Tensor,
        band: torch.Tensor,
        outside: torch.Tensor,
        dominant: torch.Tensor,
    ) -> None:
        self._bump(f"{side}_top1", int(top1.sum().item()))
        self._bump(f"{side}_topk", int(topk.sum().item()))
        self._bump(f"{side}_geo", int(geo.sum().item()))
        self._bump(f"{side}_band", int(band.sum().item()))
        self._bump(f"{side}_outside", int(outside.sum().item()))

        dom_top1 = top1 & dominant
        self._bump(f"{side}_top1_dom", int(dom_top1.sum().item()))
        self._bump(f"{side}_topk_dom", int((topk & dominant).sum().item()))
        self._bump(f"{side}_geo_dom", int((geo & dominant).sum().item()))
        self._bump(f"{side}_band_dom", int((band & dominant).sum().item()))

        per_location = dom_top1.sum(dim=1).cpu().numpy()
        target = self.per_location_rx if side == "rx" else self.per_location_tx
        target[start : start + chunk] = per_location

    def add_rx(self, start, chunk, top1, topk, geo, band, outside, dominant) -> None:
        self.total_pairs += int(dominant.numel())
        self.dominant_pairs += int(dominant.sum().item())
        self.per_location_dominant[start : start + chunk] = (
            dominant.sum(dim=1).cpu().numpy()
        )
        self._add_side("rx", start, chunk, top1, topk, geo, band, outside, dominant)

    def add_tx(self, start, chunk, top1, topk, geo, band, outside, dominant) -> None:
        # Tx quantities are per-primitive; broadcast them across the chunk so
        # the dominance mask (which IS location-dependent) can be applied.
        expand = lambda t: t.unsqueeze(0).expand(chunk, -1)
        self._add_side(
            "tx",
            start,
            chunk,
            expand(top1),
            expand(topk),
            expand(geo),
            expand(band),
            expand(outside),
            dominant,
        )

    def finalize(self, num_gaussians: int) -> Dict[str, float]:
        total = max(self.total_pairs, 1)
        dominant = max(self.dominant_pairs, 1)
        stats: Dict[str, float] = {
            "num_pairs": float(self.total_pairs),
            "num_dominant_pairs": float(self.dominant_pairs),
        }
        for key, value in self.counts.items():
            denom = dominant if key.endswith("_dom") else total
            stats[f"frac_{key}"] = value / denom
            stats[f"count_{key}"] = float(value)
        return stats


def _write_stats_csv(
    stats: Dict[str, float],
    scene,
    model_params: Namespace,
    num_locations: int,
    num_gaussians: int,
) -> None:
    path = os.path.join(OUTPUT_DIR, "wrap_stats.csv")
    rows = [
        ("run_dir", RUN_DIR),
        ("num_test_locations", num_locations),
        ("num_primitives", num_gaussians),
        ("rx_shape", f"{scene.rx_shape[0]}x{scene.rx_shape[1]}"),
        ("tx_shape", f"{scene.tx_shape[0]}x{scene.tx_shape[1]}"),
        ("k_rx", int(getattr(model_params, "max_active_rx_beams", 2))),
        ("k_tx", int(getattr(model_params, "max_active_tx_beams", 2))),
        ("dominant_fraction", DOMINANT_FRACTION),
    ]
    for key in sorted(stats):
        rows.append((key, stats[key]))

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("key,value\n")
        for key, value in rows:
            if isinstance(value, float):
                handle.write(f"{key},{value:.10g}\n")
            else:
                handle.write(f"{key},{value}\n")
    print(f"[diag_wrap] wrote {path}")


def _write_scatter(accumulator: _Accumulator, positions: torch.Tensor, scene) -> None:
    xy = positions[:, :2].cpu().numpy()
    denominator = np.maximum(accumulator.per_location_dominant, 1)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    for axis, counts, title in (
        (axes[0], accumulator.per_location_rx, "Rx side"),
        (axes[1], accumulator.per_location_tx, "Tx side"),
    ):
        # Shown as a share of that location's own dominant set: the top-10%
        # mask is defined by a quantile, and ties at the threshold make its
        # size vary from location to location.
        values = counts / denominator
        scatter = axis.scatter(
            xy[:, 0], xy[:, 1], c=values, s=6, cmap="inferno", vmin=0.0, vmax=1.0
        )
        axis.set_title(
            f"{title}: share of dominant primitives whose\n"
            f"top-1 beam changes under the wrap "
            f"(median {np.median(values):.1%}, max {values.max():.1%})"
        )
        axis.set_xlabel("x (normalized)")
        axis.set_ylabel("y (normalized)")
        axis.set_aspect("equal", adjustable="box")
        figure.colorbar(scatter, ax=axis, shrink=0.85)

    total = accumulator.per_location_rx.sum() + accumulator.per_location_tx.sum()
    figure.suptitle(
        "Raw |u-b| vs mod-2 wrapped nearest-beam disagreement, ASU test split "
        f"({len(xy)} locations, {int(total):,} dominant-primitive disagreements)"
    )
    path = os.path.join(OUTPUT_DIR, "fig_wrap_scatter.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"[diag_wrap] wrote {path}")


def _write_readme(
    stats: Dict[str, float],
    verification: List[str],
    scene,
    num_locations: int,
    num_gaussians: int,
) -> None:
    path = os.path.join(OUTPUT_DIR, "README.txt")
    lines = [AUDIT_TEXT, ""]
    lines.append("STEP 2 -- MEASURED EXPOSURE")
    lines.append("=" * 64)
    lines.append(f"checkpoint            : {CHECKPOINT}")
    lines.append(f"test locations        : {num_locations}")
    lines.append(f"primitives            : {num_gaussians}")
    lines.append(f"rx array / tx array   : {scene.rx_shape} / {scene.tx_shape}")
    lines.append(f"dominant definition   : top {DOMINANT_FRACTION:.0%} of |gain| at each location")
    lines.append("")
    lines.append("Projection reuse verification:")
    for line in verification:
        lines.append(f"  {line}")
    lines.append("")
    lines.append("Fractions over all (location, primitive) pairs:")
    for side in ("rx", "tx"):
        lines.append(f"  [{side}]")
        lines.append(f"    top-1 nearest beam disagrees      : {stats.get(f'frac_{side}_top1', 0.0):.6%}")
        lines.append(f"    top-k retained SET disagrees      : {stats.get(f'frac_{side}_topk', 0.0):.6%}")
        lines.append(f"    geometric nearest disagrees       : {stats.get(f'frac_{side}_geo', 0.0):.6%}")
        lines.append(f"    in wrap-sensitive band (u>1-1/N)  : {stats.get(f'frac_{side}_band', 0.0):.6%}")
        lines.append(f"    projected outside [-1,1]          : {stats.get(f'frac_{side}_outside', 0.0):.6%}")
    lines.append("")
    lines.append("Fractions over DOMINANT (top-10% gain) pairs only:")
    for side in ("rx", "tx"):
        lines.append(f"  [{side}]")
        lines.append(f"    top-1 nearest beam disagrees      : {stats.get(f'frac_{side}_top1_dom', 0.0):.6%}")
        lines.append(f"    top-k retained SET disagrees      : {stats.get(f'frac_{side}_topk_dom', 0.0):.6%}")
        lines.append(f"    geometric nearest disagrees       : {stats.get(f'frac_{side}_geo_dom', 0.0):.6%}")
        lines.append(f"    in wrap-sensitive band            : {stats.get(f'frac_{side}_band_dom', 0.0):.6%}")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"[diag_wrap] wrote {path}")


def _print_summary(
    stats: Dict[str, float], accumulator: _Accumulator, positions: torch.Tensor
) -> None:
    print("\n================ WRAP EXPOSURE ================")
    print(f"pairs (location x primitive) : {int(stats['num_pairs']):,}")
    print(f"dominant pairs (top 10% gain): {int(stats['num_dominant_pairs']):,}")
    for side in ("rx", "tx"):
        print(f"\n[{side}] over ALL pairs")
        print(f"  top-1 beam disagrees     : {stats.get(f'frac_{side}_top1', 0.0):.6%}")
        print(f"  top-k SET disagrees      : {stats.get(f'frac_{side}_topk', 0.0):.6%}")
        print(f"  geometric nearest differs: {stats.get(f'frac_{side}_geo', 0.0):.6%}")
        print(f"  in wrap-sensitive band   : {stats.get(f'frac_{side}_band', 0.0):.6%}")
        print(f"  outside [-1,1]           : {stats.get(f'frac_{side}_outside', 0.0):.6%}")
        print(f"[{side}] over DOMINANT pairs")
        print(f"  top-1 beam disagrees     : {stats.get(f'frac_{side}_top1_dom', 0.0):.6%}")
        print(f"  top-k SET disagrees      : {stats.get(f'frac_{side}_topk_dom', 0.0):.6%}")

    total = accumulator.per_location_rx + accumulator.per_location_tx
    order = np.argsort(-total)[:15]
    xy = positions[:, :2].cpu().numpy()
    print("\nTest locations with the most dominant-primitive disagreements:")
    print("  idx      x        y       rx     tx  total  dominant   share")
    for index in order:
        dominant = max(int(accumulator.per_location_dominant[index]), 1)
        print(
            f"  {index:5d} {xy[index,0]:8.4f} {xy[index,1]:8.4f} "
            f"{accumulator.per_location_rx[index]:6d} "
            f"{accumulator.per_location_tx[index]:6d} {total[index]:6d} "
            f"{dominant:9d} {total[index]/(2*dominant):7.1%}"
        )
    print("===============================================")


if __name__ == "__main__":
    main()
