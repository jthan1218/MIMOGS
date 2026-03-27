import torch


def magnitude_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def normalized_magnitude_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    abs_loss = magnitude_mse_loss(pred, target)
    target_power = torch.mean(target ** 2)
    return abs_loss / (target_power + eps)


def log_magnitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mag_scale: float = 1e-3,
) -> torch.Tensor:
    pred_log = torch.log1p(pred / mag_scale)
    target_log = torch.log1p(target / mag_scale)
    return torch.mean((pred_log - target_log) ** 2)


def normalize_mag_map(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (torch.amax(x) + eps)


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


def background_suppression_loss(
    pred_n: torch.Tensor,
    target_n: torch.Tensor,
    bg_threshold: float = 0.05,
) -> torch.Tensor:
    bg_mask = (target_n <= bg_threshold).float()
    denom = bg_mask.sum().clamp_min(1.0)
    return ((pred_n ** 2) * bg_mask).sum() / denom


def ranking_separation_loss(
    pred_n: torch.Tensor,
    target_n: torch.Tensor,
    topk_ratio: float = 0.125,
    num_neg: int = 16,
    margin: float = 0.05,
) -> torch.Tensor:
    pred_flat = pred_n.reshape(-1)
    target_flat = target_n.reshape(-1)

    total_num = target_flat.numel()
    k = max(1, int(round(topk_ratio * total_num)))

    pos_idx = torch.topk(target_flat, k=k, largest=True).indices

    neg_mask = torch.ones(total_num, dtype=torch.bool, device=target_flat.device)
    neg_mask[pos_idx] = False
    neg_pool_idx = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)

    if neg_pool_idx.numel() == 0:
        return pred_flat.new_zeros(())

    num_neg = min(num_neg, neg_pool_idx.numel())
    hard_order = torch.topk(pred_flat[neg_pool_idx], k=num_neg, largest=True).indices
    neg_idx = neg_pool_idx[hard_order]

    pos = pred_flat[pos_idx].unsqueeze(1)
    neg = pred_flat[neg_idx].unsqueeze(0)

    return torch.relu(margin - (pos - neg)).mean()


def hybrid_magnitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.4,
    beta: float = 0.05,
    gamma: float = 0.2,
    lambda_topk: float = 0.10,
    lambda_bg: float = 0.01,
    lambda_rank: float = 0.02,
    topk_ratio: float = 0.125,
    bg_threshold: float = 0.05,
    rank_margin: float = 0.05,
    num_neg: int = 16,
    eps: float = 1e-8,
    mag_scale: float = 0.05,
    return_terms: bool = False,
):
    pred_n = pred
    target_n = normalize_mag_map(target, eps=eps)

    abs_loss = magnitude_mse_loss(pred_n, target_n)
    rel_loss = normalized_magnitude_mse_loss(pred_n, target_n, eps=eps)
    log_loss = log_magnitude_loss(pred_n, target_n, mag_scale=mag_scale)
    topk_loss = topk_shape_loss(pred_n, target_n, topk_ratio=topk_ratio)
    bg_loss = background_suppression_loss(pred_n, target_n, bg_threshold=bg_threshold)
    rank_loss = ranking_separation_loss(
        pred_n,
        target_n,
        topk_ratio=topk_ratio,
        num_neg=num_neg,
        margin=rank_margin,
    )

    # total_loss = (
    #     0.45 * abs_loss
    #     + 0.20 * topk_loss
    #     + 0.20 * bg_loss
    #     + 0.15 * log_loss
    # )

    total_loss = 0.7 * abs_loss + 0.3 * topk_loss

    if return_terms:
        target_power = torch.mean(target_n ** 2)
        return (
            total_loss,
            abs_loss.detach(),
            rel_loss.detach(),
            log_loss.detach(),
            topk_loss.detach(),
            bg_loss.detach(),
            rank_loss.detach(),
            target_power.detach(),
        )

    return total_loss