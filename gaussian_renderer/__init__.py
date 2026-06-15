import math
from typing import Dict, Tuple, Optional

import torch

from scene.gaussian_model import GaussianModel


# def _ensure_pos_shape(x: torch.Tensor) -> torch.Tensor:
#     """Accepts shape (3,) or (1,3), returns shape (3,)"""

#     if x.dim() == 2 and x.shape[0] == 1:
#         x = x.squeeze(0)
#     if x.dim() != 1 or x.shape[0] != 3:
#         raise ValueError(f"Position must have shape (3,) or (1,3), got {tuple(x.shape)}")
#     return x

# def _assert_finite_local(name: str, x: torch.Tensor):
#     xr = torch.view_as_real(x) if torch.is_complex(x) else x
#     if not torch.isfinite(xr).all():
#         raise RuntimeError(f"[render NaN/Inf] {name}")

def _build_dft_uv_bins(num_elem: int, device, dtype) -> torch.Tensor:
    """
    Spatial-frequency bins corresponding to unshifted DFT ordering.
    For d=0.5 wavelength spacing, uv bins lie approximately in [-1,1)

    Example:
        N=4 -> [0.0, 0.5, -1.0, -0.5]
        N=2 -> [0.0, -1.0]
    """
    return 2.0 * torch.fft.fftfreq(num_elem, d=1.0, device=device).to(dtype)


_BEAM_UV_GRID_CACHE = {}

def _beam_grid_cache_key(horizontal: int, vertical: int, device, dtype):
    return (int(horizontal), int(vertical), str(device), str(dtype))

def _build_beam_uv_grid(
    horizontal: int,
    vertical: int,
    device,
    dtype,
) -> torch.Tensor:
    key = _beam_grid_cache_key(horizontal, vertical, device, dtype)

    cached = _BEAM_UV_GRID_CACHE.get(key, None)
    if cached is not None:
        return cached

    u_bins = _build_dft_uv_bins(horizontal, device=device, dtype=dtype)
    v_bins = _build_dft_uv_bins(vertical, device=device, dtype=dtype)

    u_grid = u_bins.repeat(vertical)
    v_grid = v_bins.repeat_interleave(horizontal)

    centers_uv = torch.stack([u_grid, v_grid], dim=-1)
    _BEAM_UV_GRID_CACHE[key] = centers_uv

    return centers_uv


def _direction_and_distance(
    points: torch.Tensor, # (N,3)
    array_pos: torch.Tensor # (3,)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        unit_dir: (N,3)
        dist:     (N,1)
    """
    rel = points - array_pos.unsqueeze(0)
    dist = torch.norm(rel, dim = -1, keepdim = True).clamp(min = 1e-8)
    unit_dir = rel / dist
    return unit_dir, dist


def _uv_from_unit_direction(unit_dir: torch.Tensor) -> torch.Tensor:
    """
    Convention:
    - panel plane : y-z plane
    - boresight   : +x
    - horizontal  : +y
    - vertical    : +z

    Therefore direction cosine coordinates are:
        u = d_y
        v = d_z

    Input:
        unit_dir: (N,3)
    Returns:
        uv: (N,2)
    """
    u = unit_dir[:,1]
    v = unit_dir[:,2]

    return torch.stack([u,v], dim=-1)

def _safe_inv_cov_2x2(
    cov00: torch.Tensor,
    cov01: torch.Tensor,
    cov11: torch.Tensor,
    eig_floor: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast eigenvalue-clamped inverse for symmetric 2x2 covariance matrices,
    operating on the three independent scalar components directly.

    Mathematically equivalent to:
        eigvals, eigvecs = torch.linalg.eigh([[cov00, cov01], [cov01, cov11]])
        eigvals = clamp(eigvals, min=eig_floor)
        inv = eigvecs @ diag(1/eigvals) @ eigvecs.T

    but avoids torch.linalg.eigh and matrix wrap/unwrap.
    """
    trace_half = 0.5 * (cov00 + cov11)
    diff_half = 0.5 * (cov00 - cov11)
    radius = torch.sqrt(torch.clamp(diff_half * diff_half + cov01 * cov01, min=0.0))

    lam_hi = trace_half + radius
    lam_lo = trace_half - radius

    inv_hi = 1.0 / torch.clamp(lam_hi, min=eig_floor)
    inv_lo = 1.0 / torch.clamp(lam_lo, min=eig_floor)

    # Matrix function f(C) = alpha I + beta B,
    # where C = trace_half I + B and B has eigenvalues +/- radius.
    alpha = 0.5 * (inv_hi + inv_lo)

    beta = torch.where(
        radius > 1e-12,
        0.5 * (inv_hi - inv_lo) / radius,
        torch.zeros_like(radius),
    )

    inv00 = alpha + beta * diff_half
    inv01 = beta * cov01
    inv11 = alpha - beta * diff_half

    return inv00, inv01, inv11

def _projected_angular_covariance(
    means: torch.Tensor,
    covariances: torch.Tensor,
    array_pos: torch.Tensor,
    covariance_floor: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast covariance-aware projection from 3D Gaussian to uv domain.

    Same projection as the original:
        unit_dir = (x - p) / ||x - p||
        uv = [unit_dir_y, unit_dir_z]
        Sigma_uv = J_uv Sigma_xyz J_uv^T

    but avoids materializing N×3×3 eye/Jacobian tensors.
    """
    rel = means - array_pos.unsqueeze(0)

    dist2 = (rel * rel).sum(dim=-1, keepdim=True).clamp(min=1e-16)
    inv_dist = torch.rsqrt(dist2)
    dist = torch.sqrt(dist2)

    unit_dir = rel * inv_dist
    uv_mean = unit_dir[:, 1:3]

    sx = unit_dir[:, 0]
    sy = unit_dir[:, 1]
    sz = unit_dir[:, 2]
    inv_d = inv_dist.squeeze(-1)

    # Rows of J_uv = rows 1 and 2 of (I - ss^T) / ||r||
    jy0 = (-sy * sx) * inv_d
    jy1 = (1.0 - sy * sy) * inv_d
    jy2 = (-sy * sz) * inv_d

    jz0 = (-sz * sx) * inv_d
    jz1 = (-sz * sy) * inv_d
    jz2 = (1.0 - sz * sz) * inv_d

    # Use symmetric part of covariance, matching original symmetrized output.
    c00 = covariances[:, 0, 0]
    c11 = covariances[:, 1, 1]
    c22 = covariances[:, 2, 2]
    c01 = 0.5 * (covariances[:, 0, 1] + covariances[:, 1, 0])
    c02 = 0.5 * (covariances[:, 0, 2] + covariances[:, 2, 0])
    c12 = 0.5 * (covariances[:, 1, 2] + covariances[:, 2, 1])

    def qform(a0, a1, a2):
        return (
            c00 * a0 * a0
            + c11 * a1 * a1
            + c22 * a2 * a2
            + 2.0 * c01 * a0 * a1
            + 2.0 * c02 * a0 * a2
            + 2.0 * c12 * a1 * a2
        )

    def biform(a0, a1, a2, b0, b1, b2):
        return (
            c00 * a0 * b0
            + c11 * a1 * b1
            + c22 * a2 * b2
            + c01 * (a0 * b1 + a1 * b0)
            + c02 * (a0 * b2 + a2 * b0)
            + c12 * (a1 * b2 + a2 * b1)
        )

    cov00 = qform(jy0, jy1, jy2) + covariance_floor
    cov01 = biform(jy0, jy1, jy2, jz0, jz1, jz2)
    cov11 = qform(jz0, jz1, jz2) + covariance_floor

    return uv_mean, cov00, cov01, cov11, dist

def _gaussian_beam_weights_from_uv(
    uv_mean: torch.Tensor,
    cov00: torch.Tensor,
    cov01: torch.Tensor,
    cov11: torch.Tensor,
    beam_centers_uv: torch.Tensor,
    normalize: bool = True,
    weight_floor: float = 0.0,
    eig_floor: float = 1e-4,
) -> torch.Tensor:
    inv00, inv01, inv11 = _safe_inv_cov_2x2(cov00, cov01, cov11, eig_floor=eig_floor)

    dx = beam_centers_uv[:, 0].unsqueeze(0) - uv_mean[:, 0].unsqueeze(1)
    dy = beam_centers_uv[:, 1].unsqueeze(0) - uv_mean[:, 1].unsqueeze(1)

    inv00 = inv00.unsqueeze(1)
    inv01 = inv01.unsqueeze(1)
    inv11 = inv11.unsqueeze(1)

    mahal = inv00 * dx * dx + 2.0 * inv01 * dx * dy + inv11 * dy * dy

    log_weights = torch.clamp(-0.5 * mahal, min=-80.0, max=0.0)

    weights = torch.exp(log_weights)

    if weight_floor > 0.0:
        weights = torch.where(weights < weight_floor, torch.zeros_like(weights), weights)

    if normalize:
        denom = weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        weights = weights / denom

    return weights


def _truncate_to_local_topk(
    weights: torch.Tensor,
    max_active_beams: int,
    renormalize: bool = True,
) -> torch.Tensor:
    """
    Keep only local top-k beam weights per Gaussian.

    Args:
        weights: (N, B)
        max_active_beams: maximum number of non-zero beams per Gaussian
        renormalize: if True, re-normalize kept entries to sum to 1
    """
    if weights.dim() != 2:
        raise ValueError(f"weights must have shape (N, B), got {tuple(weights.shape)}")

    num_beams = weights.shape[1]
    k = int(max_active_beams)

    if k >= num_beams:
        return weights
    if k <= 0:
        return torch.zeros_like(weights)

    topk_values, topk_indices = torch.topk(weights, k=k, dim=-1, largest=True, sorted=False)
    truncated = torch.zeros_like(weights)
    truncated.scatter_(dim=-1, index=topk_indices, src=topk_values)

    if renormalize:
        denom = truncated.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        truncated = truncated / denom

    return truncated


def render(
    rx_pos: torch.Tensor,
    tx_pos: torch.Tensor,
    pc: GaussianModel,
    rx_shape: Tuple[int, int] = (2, 2),     # (horizontal, vertical)
    tx_shape: Tuple[int, int] = (4, 4),     # (horizontal, vertical)
    scaling_modifier: float = 1.0,
    normalize_beam_weights: bool = True,
    covariance_floor: float = 1e-4,
    weight_floor: float = 0.0,
    max_active_rx_beams: int = 2,
    max_active_tx_beams: int = 2,
    renormalize_local_beam_weights: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    MIMOGS beamspace renderer.

    Output beamspace follows kron(A_y, A_x) ordering, matching MATLAB:
        A = kron(A_y, A_x)
    
    Assumptions (current v1):
    - BS / UE panel rotation = [0,0,0]
    - UPA panel lies on y-z plane
    - boresight points to +x
    - horizontal axis = +y
    - vertical axis = +z

    Inputs:
        rx_pos: (3,) or (1,3)
        tx_pos: (3,) or (1,3)
        pc    : GaussianModel

    Returns dic:
        "render"                : complex beamspace channel, shape (Nr, Nt)
        "magnitude"             : abs(render)
        "phase"                 : angle(render)
        "rx_weights"            : beam weights for receiver, shape (Nr, Nt)
        "tx_weights"            : beam weights for transmitter, shape (Nr, Nt)
        "per_Gaussian_importance": (N,)
        "beam_contributions"    : (N, Nr, Nt)
    """

    # rx_pos = _ensure_pos_shape(rx_pos).to(pc.get_xyz.device, dtype=pc.get_xyz.dtype)
    # tx_pos = _ensure_pos_shape(tx_pos).to(pc.get_xyz.device, dtype=pc.get_xyz.dtype)

    # means = pc.get_xyz      # (N,3)
    # covariances = pc.get_covariance(scaling_modifier) # (N,3,3)
    # complex_weight = pc.get_complex_weight         # (N,1) complex

    means = pc.get_xyz
    covariances = pc.get_covariance()
    gain_weight = pc.get_dynamic_gain_weight(rx_pos)
    # gain_weight = pc.get_opacity * dynamic_gain_mag
    # _assert_finite_local("gain_weight", gain_weight)

    # _assert_finite_local("means", means)
    # _assert_finite_local("covariances", covariances)

    # ------------------------------------------------------------------
    # Build beam centers in uv-domain
    # ------------------------------------------------------------------
    rx_beam_centers_uv = _build_beam_uv_grid(
        horizontal = rx_shape[0],
        vertical = rx_shape[1],
        device = means.device,
        dtype = means.dtype,
    )

    tx_beam_centers_uv = _build_beam_uv_grid(
        horizontal = tx_shape[0],
        vertical = tx_shape[1],
        device = means.device,
        dtype = means.dtype,
    )

    # ------------------------------------------------------------------
    # Covariance-aware soft projection to Rx beam-domain
    # ------------------------------------------------------------------
    rx_uv_mean, rx_cov00, rx_cov01, rx_cov11, _ = _projected_angular_covariance(
        means=means,
        covariances=covariances,
        array_pos=rx_pos,
        covariance_floor = covariance_floor,
    )

    # _assert_finite_local("rx_uv_mean", rx_uv_mean)
    # _assert_finite_local("rx_cov_uv", rx_cov_uv)


    rx_weights = _gaussian_beam_weights_from_uv(
    uv_mean=rx_uv_mean,
    cov00=rx_cov00,
    cov01=rx_cov01,
    cov11=rx_cov11,
    beam_centers_uv=rx_beam_centers_uv,
    normalize=normalize_beam_weights,
    weight_floor=weight_floor,
    eig_floor=max(covariance_floor, 1e-4),
    )
    rx_weights = _truncate_to_local_topk(
        rx_weights,
        max_active_beams=max_active_rx_beams,
        renormalize=renormalize_local_beam_weights,
    )

    # ------------------------------------------------------------------
    # Covariance-aware soft projection to Tx beam-domain
    #
    # The transmit side uses an independent 3D anchor (pc.get_xyz_tx) so the
    # tx-beam direction can be set by a different interaction point than the
    # rx-beam direction. This is what gives the model the degrees of freedom
    # to represent multi-bounce paths. When _xyz_tx == _xyz (tied init or
    # strong anchor regularization), behaviour collapses back to the original
    # single-anchor render.
    #
    # TODO(multi-bounce): plug in a separate `pc.get_covariance_tx()` here once
    # decoupled covariance is implemented. For now both projections share the
    # same per-Gaussian covariance.
    # ------------------------------------------------------------------
    means_tx = pc.get_xyz_tx
    tx_uv_mean, tx_cov00, tx_cov01, tx_cov11, _ = _projected_angular_covariance(
        means=means_tx,
        covariances=covariances,
        array_pos = tx_pos,
        covariance_floor = covariance_floor,
    )
    tx_weights = _gaussian_beam_weights_from_uv(
        uv_mean=tx_uv_mean,
        cov00=tx_cov00,
        cov01=tx_cov01,
        cov11=tx_cov11,
        beam_centers_uv=tx_beam_centers_uv,
        normalize=normalize_beam_weights,
        weight_floor=weight_floor,
        eig_floor=max(covariance_floor, 1e-4),
    )
    tx_weights = _truncate_to_local_topk(
        tx_weights,
        max_active_beams=max_active_tx_beams,
        renormalize=renormalize_local_beam_weights,
    )

    # ------------------------------------------------------------------
    # Beamspace splatting / superposition
    # H_n[p,q] = c_n * r_n[p] * t_n[q]
    # ------------------------------------------------------------------ 
    gain = gain_weight.reshape(-1)  # (N,)

    H = rx_weights.transpose(0, 1) @ (tx_weights * gain[:, None])

    with torch.no_grad():
        per_gaussian_importance = (
            gain.detach().abs()
            * rx_weights.detach().sum(dim=1)
            * tx_weights.detach().sum(dim=1)
        )

    return {
        "render": H,
        "magnitude": H,
        "rx_weights": rx_weights,
        "tx_weights": tx_weights,
        "per_gaussian_importance": per_gaussian_importance,
        "gain_weight": gain_weight,
    }