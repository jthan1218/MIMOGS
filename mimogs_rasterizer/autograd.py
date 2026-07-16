"""Autograd wrapper for the optional fused CUDA rasterizer."""
from __future__ import annotations

import torch

try:
    import mimogs_rasterizer_cuda as _C
except ImportError:
    _C = None


def cuda_extension_available() -> bool:
    return _C is not None


class _BeamSplatCUDA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rx_uv: torch.Tensor,
        rx_precision: torch.Tensor,
        tx_uv: torch.Tensor,
        tx_precision: torch.Tensor,
        gain: torch.Tensor,
        rx_centers: torch.Tensor,
        tx_centers: torch.Tensor,
        k_rx: int,
        k_tx: int,
        weight_floor: float,
    ) -> torch.Tensor:
        if _C is None:
            raise RuntimeError(
                "mimogs_rasterizer_cuda is not installed. Run "
                "`pip install -v ./mimogs_rasterizer` from the project root."
            )

        tensors = (
            rx_uv.contiguous(),
            rx_precision.contiguous(),
            tx_uv.contiguous(),
            tx_precision.contiguous(),
            gain.contiguous(),
            rx_centers.contiguous(),
            tx_centers.contiguous(),
        )
        output, rx_idx, rx_w, tx_idx, tx_w = _C.forward(
            *tensors, int(k_rx), int(k_tx), float(weight_floor)
        )
        ctx.save_for_backward(
            tensors[0],
            tensors[1],
            tensors[2],
            tensors[3],
            tensors[4],
            tensors[5],
            tensors[6],
            rx_idx,
            rx_w,
            tx_idx,
            tx_w,
        )
        ctx.k_rx = int(k_rx)
        ctx.k_tx = int(k_tx)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            rx_uv,
            rx_precision,
            tx_uv,
            tx_precision,
            gain,
            rx_centers,
            tx_centers,
            rx_idx,
            rx_w,
            tx_idx,
            tx_w,
        ) = ctx.saved_tensors

        grad_rx_uv, grad_rx_precision, grad_tx_uv, grad_tx_precision, grad_gain = _C.backward(
            grad_output.contiguous(),
            rx_uv,
            rx_precision,
            tx_uv,
            tx_precision,
            gain,
            rx_centers,
            tx_centers,
            rx_idx,
            rx_w,
            tx_idx,
            tx_w,
        )

        return (
            grad_rx_uv,
            grad_rx_precision,
            grad_tx_uv,
            grad_tx_precision,
            grad_gain,
            None,
            None,
            None,
            None,
            None,
        )


def beam_splat_cuda(
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
) -> torch.Tensor:
    return _BeamSplatCUDA.apply(
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
