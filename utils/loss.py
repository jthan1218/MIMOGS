import torch


def magnitude_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def normalize_mag_map(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize each map independently by its maximum entry."""
    if x.dim() <= 2:
        return x / torch.amax(x).clamp_min(eps)
    reduce_dims = tuple(range(1, x.dim()))
    scale = torch.amax(x, dim=reduce_dims, keepdim=True).clamp_min(eps)
    return x / scale


def magnitude_nmse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    if pred.dim() == 2:
        num = torch.sum((pred - target) ** 2)
        den = torch.sum(target ** 2).clamp_min(eps)
        return num / den

    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    num = torch.sum((pred_flat - target_flat) ** 2, dim=1)
    den = torch.sum(target_flat ** 2, dim=1).clamp_min(eps)
    return torch.mean(num / den)


def topk_shape_loss(
    pred_n: torch.Tensor,
    target_n: torch.Tensor,
    topk_ratio: float = 0.125,
) -> torch.Tensor:
    """Per-map top-k reconstruction loss (batch-safe)."""
    single = pred_n.dim() == 2
    if single:
        pred_flat = pred_n.reshape(1, -1)
        target_flat = target_n.reshape(1, -1)
    else:
        pred_flat = pred_n.reshape(pred_n.shape[0], -1)
        target_flat = target_n.reshape(target_n.shape[0], -1)

    k = max(1, int(round(topk_ratio * target_flat.shape[1])))
    topk_idx = torch.topk(target_flat, k=k, dim=1, largest=True).indices
    pred_topk = torch.gather(pred_flat, dim=1, index=topk_idx)
    target_topk = torch.gather(target_flat, dim=1, index=topk_idx)
    return torch.mean((pred_topk - target_topk) ** 2)


def hybrid_magnitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    topk_ratio: float = 0.125,
    eps: float = 1e-8,
    return_terms: bool = False,
):
    # The draft code trains on a location-wise max-normalized target.  The
    # renderer's output is already in that learned scale and is not normalized
    # here, preserving the original training objective.
    pred_n = pred
    target_n = normalize_mag_map(target, eps=eps)

    abs_loss = magnitude_nmse_loss(pred_n, target_n, eps=eps)
    topk_loss = topk_shape_loss(pred_n, target_n, topk_ratio=topk_ratio)
    total_loss = 0.7 * abs_loss + 0.3 * topk_loss

    if return_terms:
        return total_loss, abs_loss.detach(), topk_loss.detach()
    return total_loss
