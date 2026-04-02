import torch


def magnitude_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)

def normalize_mag_map(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (torch.amax(x) + eps)


# def topk_shape_loss(
#     pred_n: torch.Tensor,
#     target_n: torch.Tensor,
#     # topk_ratio: float = 0.125,
#     topk_ratio: float = 0.0625,
# ) -> torch.Tensor:
#     pred_flat = pred_n.reshape(-1)
#     target_flat = target_n.reshape(-1)

#     weight_power = 1.5
#     eps = 1e-8

#     k = max(1, int(round(topk_ratio * target_flat.numel())))
#     topk_idx = torch.topk(target_flat, k=k, largest=True).indices

#     pred_topk = pred_flat[topk_idx]
#     target_topk = target_flat[topk_idx]

#     weights = target_topk.clamp_min(eps).pow(weight_power)
#     weights = weights / weights.sum().clamp_min(eps)

#     return torch.sum(weights * (pred_topk - target_topk) ** 2)

def topk_shape_loss(
    pred_n: torch.Tensor,
    target_n: torch.Tensor,
    topk_ratio: float = 0.125,
) -> torch.Tensor:
    pred_flat = pred_n.reshape(-1)
    target_flat = target_n.reshape(-1)
    k = max(1, int(round(topk_ratio * target_flat.numel())))
    topk_idx = torch.topk(target_flat, k=k, largest=True).indices
    return torch.mean((pred_flat[topk_idx] - target_flat[topk_idx]) ** 2)


def hybrid_magnitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    topk_ratio: float = 0.125,
    eps: float = 1e-8,
    return_terms: bool = False,
):
    pred_n = pred
    target_n = normalize_mag_map(target, eps=eps)

    abs_loss = magnitude_mse_loss(pred_n, target_n)
    topk_loss = topk_shape_loss(pred_n, target_n, topk_ratio=topk_ratio)

    total_loss = 0.7 * abs_loss + 0.3 * topk_loss

    if return_terms:
        # target_power = torch.mean(target_n ** 2)
        return (
            total_loss,
            abs_loss.detach(),
            topk_loss.detach(),
            # target_power.detach(),
        )

    return total_loss