"""Batched MIMO-GS renderer with sparse beam-pair splatting.

This module preserves the mathematical renderer in the draft while removing
three avoidable costs in the original implementation:

1. Tx-side projection is evaluated once per batch (the BS is fixed).
2. Only K_r and K_t retained beam weights are carried into splatting.
3. Multiple UE query locations are rendered in one call.

When the optional CUDA extension is installed, Gaussian beam evaluation,
top-k selection, normalization, and sparse accumulation are fused.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from mimogs_rasterizer import beam_splat
from scene.gaussian_model import GaussianModel


_BEAM_UV_GRID_CACHE = {}


def _build_dft_uv_bins(num_elem: int, device, dtype) -> torch.Tensor:
    return 2.0 * torch.fft.fftfreq(num_elem, d=1.0, device=device).to(dtype)


def _build_beam_uv_grid(
    horizontal: int,
    vertical: int,
    device,
    dtype,
) -> torch.Tensor:
    key = (int(horizontal), int(vertical), str(device), str(dtype))
    cached = _BEAM_UV_GRID_CACHE.get(key)
    if cached is not None:
        return cached

    u_bins = _build_dft_uv_bins(horizontal, device, dtype)
    v_bins = _build_dft_uv_bins(vertical, device, dtype)
    grid = torch.stack(
        [u_bins.repeat(vertical), v_bins.repeat_interleave(horizontal)], dim=-1
    ).contiguous()
    _BEAM_UV_GRID_CACHE[key] = grid
    return grid


def _as_position_batch(x: torch.Tensor, device, dtype) -> Tuple[torch.Tensor, bool]:
    x = x.to(device=device, dtype=dtype)
    if x.dim() == 1:
        if x.numel() != 3:
            raise ValueError(f"position must have 3 entries, got {tuple(x.shape)}")
        return x.view(1, 3), True
    if x.dim() == 2 and x.shape[-1] == 3:
        return x, False
    raise ValueError(f"position must have shape (3,) or (B,3), got {tuple(x.shape)}")


def _safe_inv_cov_2x2(
    cov00: torch.Tensor,
    cov01: torch.Tensor,
    cov11: torch.Tensor,
    eig_floor: float = 1e-4,
) -> torch.Tensor:
    """Return precision components ``(p00,p01,p11)`` without an eigendecomp."""
    trace_half = 0.5 * (cov00 + cov11)
    diff_half = 0.5 * (cov00 - cov11)

    radius = torch.sqrt((diff_half.square() + cov01.square()).clamp_min(1e-12))

    lam_hi = trace_half + radius
    lam_lo = trace_half - radius
    inv_hi = torch.clamp(lam_hi, min=eig_floor).reciprocal()
    inv_lo = torch.clamp(lam_lo, min=eig_floor).reciprocal()

    alpha = 0.5 * (inv_hi + inv_lo)
    beta = 0.5 * (inv_hi - inv_lo) / radius
    return torch.stack(
        [
            alpha + beta * diff_half,
            beta * cov01,
            alpha - beta * diff_half,
        ],
        dim=-1,
    )


def _projected_angular_covariance_batched(
    means: torch.Tensor,
    covariances: torch.Tensor,
    array_positions: torch.Tensor,
    covariance_floor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project N 3-D Gaussians for B array locations.

    Args:
        means: ``(N,3)``.
        covariances: ``(N,3,3)``.
        array_positions: ``(B,3)``.

    Returns:
        projected means ``(B,N,2)`` and precision components ``(B,N,3)``.
    """
    rel = means.unsqueeze(0) - array_positions.unsqueeze(1)  # (B,N,3)
    dist2 = rel.square().sum(dim=-1, keepdim=True).clamp_min(1e-16)
    inv_dist = torch.rsqrt(dist2)
    unit = rel * inv_dist

    sx, sy, sz = unit.unbind(dim=-1)
    inv_d = inv_dist.squeeze(-1)

    jy0 = (-sy * sx) * inv_d
    jy1 = (1.0 - sy.square()) * inv_d
    jy2 = (-sy * sz) * inv_d
    jz0 = (-sz * sx) * inv_d
    jz1 = (-sz * sy) * inv_d
    jz2 = (1.0 - sz.square()) * inv_d

    c00 = covariances[:, 0, 0].unsqueeze(0)
    c11 = covariances[:, 1, 1].unsqueeze(0)
    c22 = covariances[:, 2, 2].unsqueeze(0)
    c01 = (0.5 * (covariances[:, 0, 1] + covariances[:, 1, 0])).unsqueeze(0)
    c02 = (0.5 * (covariances[:, 0, 2] + covariances[:, 2, 0])).unsqueeze(0)
    c12 = (0.5 * (covariances[:, 1, 2] + covariances[:, 2, 1])).unsqueeze(0)

    def qform(a0, a1, a2):
        return (
            c00 * a0.square()
            + c11 * a1.square()
            + c22 * a2.square()
            + 2.0 * c01 * a0 * a1
            + 2.0 * c02 * a0 * a2
            + 2.0 * c12 * a1 * a2
        )

    cov00 = qform(jy0, jy1, jy2) + float(covariance_floor)
    cov11 = qform(jz0, jz1, jz2) + float(covariance_floor)
    cov01 = (
        c00 * jy0 * jz0
        + c11 * jy1 * jz1
        + c22 * jy2 * jz2
        + c01 * (jy0 * jz1 + jy1 * jz0)
        + c02 * (jy0 * jz2 + jy2 * jz0)
        + c12 * (jy1 * jz2 + jy2 * jz1)
    )

    uv = unit[..., 1:3].contiguous()
    precision = _safe_inv_cov_2x2(
        cov00, cov01, cov11, eig_floor=max(float(covariance_floor), 1e-4)
    ).contiguous()
    return uv, precision


def render_fast(
    rx_pos: torch.Tensor,
    tx_pos: torch.Tensor,
    pc: GaussianModel,
    rx_shape: Tuple[int, int] = (2, 2),
    tx_shape: Tuple[int, int] = (4, 4),
    covariance_floor: float = 1e-4,
    weight_floor: float = 1e-4,
    max_active_rx_beams: int = 2,
    max_active_tx_beams: int = 2,
    use_cuda_rasterizer: bool = True,
) -> Dict[str, torch.Tensor]:
    """Render one or more UE locations.

    ``rx_pos`` may be ``(3,)`` or ``(B,3)``.  For a single input, ``render`` is
    returned as ``(N_r,N_t)`` to remain drop-in compatible with the old API;
    otherwise it is ``(B,N_r,N_t)``.
    """
    means_rx = pc.get_xyz
    device, dtype = means_rx.device, means_rx.dtype
    rx_batch, squeeze_output = _as_position_batch(rx_pos, device, dtype)
    tx_batch, _ = _as_position_batch(tx_pos, device, dtype)
    if tx_batch.shape[0] != 1:
        raise ValueError("The current fixed-BS renderer expects one Tx position")

    covariances = pc.get_covariance()
    # Keep the geometric rasterizer in FP32 even when the gain MLP is
    # evaluated under autocast. The cast is differentiable.
    gain = pc.get_dynamic_gain_weight_batched(rx_batch).to(dtype=dtype)  # (B,N)

    rx_uv, rx_precision = _projected_angular_covariance_batched(
        means_rx, covariances, rx_batch, covariance_floor
    )

    rx_uv = - rx_uv

    # The BS and Tx-side anchors are fixed, so this is evaluated only once for
    # the entire query batch. The Tx side carries its own 3D covariance, so the
    # resulting tx_precision below is distinct from rx_precision; the two ends
    # of a primitive are tied only through the shared gain.
    tx_uv_b, tx_precision_b = _projected_angular_covariance_batched(
        pc.get_xyz_tx, pc.get_covariance_tx(), tx_batch, covariance_floor
    )
    tx_uv = tx_uv_b.squeeze(0).contiguous()
    tx_precision = tx_precision_b.squeeze(0).contiguous()

    rx_centers = _build_beam_uv_grid(
        rx_shape[0], rx_shape[1], device=device, dtype=dtype
    )
    tx_centers = _build_beam_uv_grid(
        tx_shape[0], tx_shape[1], device=device, dtype=dtype
    )

    output = beam_splat(
        rx_uv=rx_uv,
        rx_precision=rx_precision,
        tx_uv=tx_uv,
        tx_precision=tx_precision,
        gain=gain,
        rx_centers=rx_centers,
        tx_centers=tx_centers,
        k_rx=min(int(max_active_rx_beams), int(rx_centers.shape[0])),
        k_tx=min(int(max_active_tx_beams), int(tx_centers.shape[0])),
        weight_floor=float(weight_floor),
        use_cuda_extension=bool(use_cuda_rasterizer),
    )

    # Because retained Rx/Tx weights are normalized per Gaussian, the total
    # mass contributed by a Gaussian is its gain.  This is also a useful
    # pruning statistic and requires no materialized dense weight matrices.
    importance = gain.detach().abs()

    if squeeze_output:
        output = output.squeeze(0)
        importance = importance.squeeze(0)

    return {
        "render": output,
        "magnitude": output,
        "per_gaussian_importance": importance,
        "gain_weight": gain.squeeze(0).unsqueeze(-1) if squeeze_output else gain,
    }


# Optional drop-in name for callers that import ``render`` from this module.
render = render_fast
