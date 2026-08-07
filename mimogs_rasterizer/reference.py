"""Reference sparse beam-pair rasterizer implemented with PyTorch ops.

The implementation is fully differentiable and supports CPU/CUDA.  It is used
both as a correctness reference and as a fallback when the optional CUDA
extension is not built.
"""
from __future__ import annotations

from typing import Tuple

import torch


def wrap_beam_delta(delta: torch.Tensor) -> torch.Tensor:
    """Wrap a beam-coordinate difference into ``[-1, 1)``.

    Beam centres live on ``2 * fftfreq(N)`` and the array response phase is
    ``exp(j*pi*k*(u - b))``, which is periodic in ``u - b`` with period 2.  The
    raw difference therefore over-states the distance for coordinates near the
    grid edge: with ``N=4`` a mean at ``u = 0.9`` is 1.9 away from the bin at
    ``-1.0`` under the raw metric but only 0.1 away under the true one, so the
    unwrapped renderer picks the wrong beam.

    The map is piecewise identity with unit slope, so gradients are unchanged
    away from the (measure-zero) branch cut.

    This is DFT-specific.  A measured analog steering codebook has bounded,
    non-periodic beam centres, so wrapping would fold distant beams onto each
    other; those callers pass ``periodic=False`` and use the raw difference.
    """
    return torch.remainder(delta + 1.0, 2.0) - 1.0


def topk_gaussian_beam_weights(
    uv_mean: torch.Tensor,
    precision: torch.Tensor,
    beam_centers_uv: torch.Tensor,
    k: int,
    weight_floor: float = 0.0,
    periodic: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate an anisotropic Gaussian only on its top-k beam bins.

    Args:
        uv_mean: ``(..., 2)`` projected means.
        precision: ``(..., 3)`` storing ``(p00, p01, p11)``.
        beam_centers_uv: ``(num_beams, 2)``.
        k: number of retained beams; must be positive.
        weight_floor: optional floor applied to the selected weights relative
            to the strongest selected beam (whose relative weight is one), so
            negligible beams can be dropped but the row can never collapse to
            all zeros.
        periodic: whether the beam centres are a DFT grid, i.e. period-2 in the
            beam coordinate.  ``False`` (a measured steering codebook) uses the
            raw difference -- see ``wrap_beam_delta``.

    Returns:
        normalized top-k values and int64 beam indices, each with shape
        ``(..., min(k, num_beams))``.

    Top-k indices are piecewise constant.  This matches ``torch.topk`` in the
    original implementation: gradients are propagated through selected values,
    not through changes in the selected indices.
    """
    if uv_mean.shape[-1] != 2:
        raise ValueError(f"uv_mean must end in dimension 2, got {uv_mean.shape}")
    if precision.shape[:-1] != uv_mean.shape[:-1] or precision.shape[-1] != 3:
        raise ValueError(
            "precision must have shape uv_mean.shape[:-1] + (3,), got "
            f"{precision.shape} for uv_mean {uv_mean.shape}"
        )
    if beam_centers_uv.dim() != 2 or beam_centers_uv.shape[-1] != 2:
        raise ValueError("beam_centers_uv must have shape (num_beams, 2)")

    num_beams = int(beam_centers_uv.shape[0])
    k_eff = min(int(k), num_beams)
    if k_eff <= 0:
        raise ValueError(f"k must be positive, got {k}")

    # Wrapped modulo 2 on the DFT grid: this feeds both the weight exponent
    # and, through the logits, the top-k candidate selection below.  A custom
    # steering codebook is non-periodic and uses the raw difference.
    delta_x = beam_centers_uv[:, 0] - uv_mean[..., 0, None]
    delta_y = beam_centers_uv[:, 1] - uv_mean[..., 1, None]
    dx = wrap_beam_delta(delta_x) if periodic else delta_x
    dy = wrap_beam_delta(delta_y) if periodic else delta_y

    p00 = precision[..., 0, None]
    p01 = precision[..., 1, None]
    p11 = precision[..., 2, None]

    mahal = p00 * dx.square() + 2.0 * p01 * dx * dy + p11 * dy.square()
    # Top-k must run on the raw logits: clamping first flattens distant beams
    # to a common floor and makes the selection order arbitrary.
    logits = -0.5 * mahal

    top_logits, top_indices = torch.topk(
        logits, k=k_eff, dim=-1, largest=True, sorted=False
    )

    # Stable softmax over the selected beams: the strongest selected logit is
    # shifted to zero, so its weight is exp(0)=1 and the normalized weights
    # always sum to one instead of collapsing when every logit is very small.
    shifted_logits = top_logits - top_logits.amax(dim=-1, keepdim=True).detach()
    top_weights = torch.exp(shifted_logits)

    if weight_floor > 0.0:
        top_weights = torch.where(
            top_weights >= float(weight_floor),
            top_weights,
            torch.zeros_like(top_weights),
        )

    denom = top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    top_weights = top_weights / denom
    return top_weights, top_indices


def sparse_outer_splat_reference(
    rx_weights: torch.Tensor,
    rx_indices: torch.Tensor,
    tx_weights: torch.Tensor,
    tx_indices: torch.Tensor,
    gain: torch.Tensor,
    num_rx_beams: int,
    num_tx_beams: int,
) -> torch.Tensor:
    """Accumulate sparse separable Gaussian contributions.

    Args:
        rx_weights/indices: ``(B, N, K_r)``.
        tx_weights/indices: ``(N, K_t)``; Tx projection is shared by all query
            locations because the BS and Tx-side anchors are fixed.
        gain: ``(B, N)``.

    Returns:
        Beam-pair power maps of shape ``(B, N_r, N_t)``.
    """
    if rx_weights.dim() != 3:
        raise ValueError("rx_weights must have shape (B, N, K_r)")
    if tx_weights.dim() != 2:
        raise ValueError("tx_weights must have shape (N, K_t)")
    if gain.shape != rx_weights.shape[:2]:
        raise ValueError(
            f"gain must have shape {rx_weights.shape[:2]}, got {gain.shape}"
        )
    if rx_indices.shape != rx_weights.shape or tx_indices.shape != tx_weights.shape:
        raise ValueError("weight/index shape mismatch")
    if tx_weights.shape[0] != rx_weights.shape[1]:
        raise ValueError("Rx and Tx tensors must have the same Gaussian count")

    batch_size, num_gaussians, k_rx = rx_weights.shape
    k_tx = tx_weights.shape[-1]
    num_pairs = int(num_rx_beams) * int(num_tx_beams)

    output = gain.new_zeros((batch_size, num_pairs))
    tx_indices_b = tx_indices.unsqueeze(0).expand(batch_size, -1, -1)
    tx_weights_b = tx_weights.unsqueeze(0)

    # K_r and K_t are intentionally small (typically 2).  Using only K_r*K_t
    # scatter-adds avoids materializing N x N_r and N x N_t zero-filled arrays.
    for i in range(k_rx):
        r_idx = rx_indices[..., i]
        r_weight = rx_weights[..., i]
        for j in range(k_tx):
            flat_idx = r_idx * int(num_tx_beams) + tx_indices_b[..., j]
            src = gain * r_weight * tx_weights_b[..., j]
            output.scatter_add_(dim=1, index=flat_idx, src=src)

    return output.view(batch_size, int(num_rx_beams), int(num_tx_beams))


def beam_splat_reference(
    rx_uv: torch.Tensor,
    rx_precision: torch.Tensor,
    tx_uv: torch.Tensor,
    tx_precision: torch.Tensor,
    gain: torch.Tensor,
    rx_centers: torch.Tensor,
    tx_centers: torch.Tensor,
    k_rx: int,
    k_tx: int,
    weight_floor: float = 0.0,
    return_aux: bool = False,
    periodic: bool = True,
):
    """Complete differentiable reference beam-pair rasterizer."""
    rx_weights, rx_indices = topk_gaussian_beam_weights(
        rx_uv, rx_precision, rx_centers, k_rx, weight_floor, periodic
    )
    tx_weights, tx_indices = topk_gaussian_beam_weights(
        tx_uv, tx_precision, tx_centers, k_tx, weight_floor, periodic
    )
    output = sparse_outer_splat_reference(
        rx_weights,
        rx_indices,
        tx_weights,
        tx_indices,
        gain,
        rx_centers.shape[0],
        tx_centers.shape[0],
    )
    if return_aux:
        return output, {
            "rx_weights": rx_weights,
            "rx_indices": rx_indices,
            "tx_weights": tx_weights,
            "tx_indices": tx_indices,
        }
    return output
