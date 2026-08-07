"""MIMO-GS sparse beam-pair rasterizer.

The package exposes a differentiable PyTorch reference path and, when built,
a fused CUDA path.  The CUDA operator is deliberately limited to the exact
mode used by the draft code: float32 tensors, top-k support, and per-Gaussian
normalization of retained beam weights.
"""
from __future__ import annotations

from typing import Optional

import torch

from .reference import beam_splat_reference

try:
    from .autograd import beam_splat_cuda, cuda_extension_available
except Exception:  # Import must never make the reference renderer unusable.
    beam_splat_cuda = None

    def cuda_extension_available() -> bool:
        return False


def beam_splat(
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
    use_cuda_extension: bool = True,
    periodic: bool = True,
):
    """Rasterize a batch of beam-pair power maps.

    Falls back to the fully differentiable PyTorch implementation when the
    extension is unavailable or when any input is not a CUDA float32 tensor.

    ``periodic=False`` (custom, non-DFT beam centres) also forces the reference
    path: the CUDA kernel wraps beam deltas modulo 2 unconditionally, which is
    only correct for a DFT grid.
    """
    can_use_cuda = (
        use_cuda_extension
        and periodic
        and beam_splat_cuda is not None
        and cuda_extension_available()
        and rx_uv.is_cuda
        and rx_uv.dtype == torch.float32
        and all(
            t.is_cuda and t.dtype == torch.float32
            for t in (
                rx_precision,
                tx_uv,
                tx_precision,
                gain,
                rx_centers,
                tx_centers,
            )
        )
    )
    if can_use_cuda:
        return beam_splat_cuda(
            rx_uv,
            rx_precision,
            tx_uv,
            tx_precision,
            gain,
            rx_centers,
            tx_centers,
            int(k_rx),
            int(k_tx),
            float(weight_floor),
        )

    return beam_splat_reference(
        rx_uv,
        rx_precision,
        tx_uv,
        tx_precision,
        gain,
        rx_centers,
        tx_centers,
        int(k_rx),
        int(k_tx),
        float(weight_floor),
        periodic=bool(periodic),
    )


__all__ = [
    "beam_splat",
    "beam_splat_reference",
    "cuda_extension_available",
]
