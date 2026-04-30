import math
from typing import Dict, Tuple, Optional

import torch

from scene.gaussian_model import GaussianModel


def _ensure_pos_shape(x: torch.Tensor) -> torch.Tensor:
    """Accepts shape (3,) or (1,3), returns shape (3,)"""

    if x.dim() == 2 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.dim() != 1 or x.shape[0] != 3:
        raise ValueError(f"Position must have shape (3,) or (1,3), got {tuple(x.shape)}")
    return x

# def _assert_finite_local(name: str, x: torch.Tensor):
#     xr = torch.view_as_real(x) if torch.is_complex(x) else x
#     if not torch.isfinite(xr).all():
#         raise RuntimeError(f"[render NaN/Inf] {name}")

def _symmetrize(mat: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mat + mat.transpose(-1, -2))

def _build_dft_uv_bins(num_elem: int, device, dtype) -> torch.Tensor:
    """
    Spatial-frequency bins corresponding to unshifted DFT ordering.
    For d=0.5 wavelength spacing, uv bins lie approximately in [-1,1)

    Example:
        N=4 -> [0.0, 0.5, -1.0, -0.5]
        N=2 -> [0.0, -1.0]
    """
    return 2.0 * torch.fft.fftfreq(num_elem, d=1.0, device=device).to(dtype)


def _build_beam_uv_grid(
    horizontal: int,
    vertical: int,
    device,
    dtype,
) -> torch.Tensor:
    """
    Build beam-center grid in uv domain.

    Ordering matches kron(A_y, A_x):
        fast index = horizontal
        slow index = vertical

    Returns:
        centers_uv: (vertical, horizontal, 2)
                    columns are [u_horizontal, v_vertical]
    """

    u_bins = _build_dft_uv_bins(horizontal, device = device, dtype = dtype) # x/horizontal fast
    v_bins = _build_dft_uv_bins(vertical, device = device, dtype = dtype)   # y/vertical slow

    u_grid = u_bins.repeat(vertical)
    v_grid = v_bins.repeat_interleave(horizontal)

    centers_uv = torch.stack([u_grid, v_grid], dim = -1)
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

def _projected_angular_covariance(
    means: torch.Tensor,            # (N,3)
    covariances: torch.Tensor,      # (N,3,3)
    array_pos: torch.Tensor,        # (3,)
    covariance_floor: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Covariance-aware projection from 3D Gaussian to uv domain.

    Uses first-order projection:
        unit_dir = (x - p) / ||x - p||
        uv = [unit_dir_y, unit_dir, z]
        Sigma_uv = J_uv Sigma_xyz J_uv^T

    where J_uv is the Jacobian of uv w.r.t xyz evaluated at the mean.

    Returns:
        uv_mean: (N,2)
        cov_uv:  (N,2,2)
        dist:    (N,1)
    """

    device = means.device
    dtype = means.dtype
    N = means.shape[0]
    # _assert_finite_local("means", means)
    # _assert_finite_local("covariances", covariances)
    # _assert_finite_local("array_pos", array_pos)

    unit_dir, dist = _direction_and_distance(means, array_pos)      # (N,3), (N,1)
    uv_mean = _uv_from_unit_direction(unit_dir)                     # (N,2)

    # _assert_finite_local("unit_dir", unit_dir)
    # _assert_finite_local("dist", dist)
    # _assert_finite_local("uv_mean", uv_mean)


    # Jacobian of normalized vector: J = (I - uu^T) / ||r||
    eye3 = torch.eye(3, device = device, dtype = dtype).unsqueeze(0).expand(means.shape[0], -1, -1)
    uuT = unit_dir.unsqueeze(-1) @ unit_dir.unsqueeze(-2)           # (N,3,3)
    J_unit = (eye3 - uuT) / dist.unsqueeze(-1)                      # (N,3,3)

    # uv = [unit_dir_y, unit_dir_z], so keep rows 1 and 2
    J_uv = J_unit[:, 1:3, :]                                       # (N,2,3)
    # _assert_finite_local("J_uv", J_uv)

    cov_uv = J_uv @ covariances @ J_uv.transpose(-1, -2)           # (N,2,2)
    cov_uv = _symmetrize(cov_uv)

    eye2 = torch.eye(2, device=device, dtype=dtype).unsqueeze(0).expand(means.shape[0], -1, -1)
    cov_uv = cov_uv + covariance_floor * eye2
    
    # _assert_finite_local("cov_uv", cov_uv)
    return uv_mean, cov_uv, dist

def _safe_inv_cov_2x2(
    cov_uv: torch.Tensor,
    eig_floor: float = 1e-4,
) -> torch.Tensor:
    """
    Stable inverse for symmetric 2x2 covariance matrices using eigendecomposition.
    """
    cov_uv = _symmetrize(cov_uv)
    eigvals, eigvecs = torch.linalg.eigh(cov_uv)  # eigvals ascending

    # _assert_finite_local("cov_uv_eigvals", eigvals)
    # _assert_finite_local("cov_uv_eigvecs", eigvecs)

    eigvals = torch.clamp(eigvals, min=eig_floor)
    inv_eigvals = 1.0 / eigvals
    inv_cov_uv = eigvecs @ torch.diag_embed(inv_eigvals) @ eigvecs.transpose(-1, -2)
    inv_cov_uv = _symmetrize(inv_cov_uv)

    # _assert_finite_local("inv_cov_uv", inv_cov_uv)
    return inv_cov_uv

def _gaussian_beam_weights_from_uv(
    uv_mean: torch.Tensor,
    cov_uv: torch.Tensor,
    beam_centers_uv: torch.Tensor,
    normalize: bool = True,
    weight_floor: float = 0.0,
    eig_floor: float = 1e-4,
) -> torch.Tensor:
    # _assert_finite_local("uv_mean", uv_mean)
    # _assert_finite_local("cov_uv_input", cov_uv)
    # _assert_finite_local("beam_centers_uv", beam_centers_uv)

    delta = beam_centers_uv.unsqueeze(0) - uv_mean.unsqueeze(1)
    # _assert_finite_local("delta", delta)

    inv_cov_uv = _safe_inv_cov_2x2(cov_uv, eig_floor=eig_floor)

    mahal = torch.einsum("nbi,nij,nbj->nb", delta, inv_cov_uv, delta)
    # _assert_finite_local("mahal", mahal)

    log_weights = torch.clamp(-0.5 * mahal, min=-80.0, max=0.0)
    weights = torch.exp(log_weights)

    if weight_floor > 0.0:
        weights = torch.where(weights < weight_floor, torch.zeros_like(weights), weights)

    if normalize:
        denom = weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        weights = weights / denom

    # _assert_finite_local("weights", weights)
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

    rx_pos = _ensure_pos_shape(rx_pos).to(pc.get_xyz.device, dtype=pc.get_xyz.dtype)
    tx_pos = _ensure_pos_shape(tx_pos).to(pc.get_xyz.device, dtype=pc.get_xyz.dtype)

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
    rx_uv_mean, rx_cov_uv, _ = _projected_angular_covariance(
        means=means,
        covariances=covariances,
        array_pos=rx_pos,
        covariance_floor = covariance_floor,
    )

    # _assert_finite_local("rx_uv_mean", rx_uv_mean)
    # _assert_finite_local("rx_cov_uv", rx_cov_uv)


    rx_weights = _gaussian_beam_weights_from_uv(
    uv_mean=rx_uv_mean,
    cov_uv=rx_cov_uv,
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

    # _assert_finite_local("rx_weights", rx_weights)

    # ------------------------------------------------------------------
    # Covariance-aware soft projection to Tx beam-domain
    # ------------------------------------------------------------------
    tx_uv_mean, tx_cov_uv, _ = _projected_angular_covariance(
        means=means,
        covariances=covariances,
        array_pos = tx_pos,
        covariance_floor = covariance_floor,
    )

    # _assert_finite_local("tx_uv_mean", tx_uv_mean)
    # _assert_finite_local("tx_cov_uv", tx_cov_uv)

    tx_weights = _gaussian_beam_weights_from_uv(
        uv_mean=tx_uv_mean,
        cov_uv=tx_cov_uv,
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

    # _assert_finite_local("tx_weights", tx_weights)

    # ------------------------------------------------------------------
    # Beamspace splatting / superposition
    # H_n[p,q] = c_n * r_n[p] * t_n[q]
    # ------------------------------------------------------------------ 
    beam_contributions = (
        gain_weight.view(-1, 1, 1)
        * rx_weights[:, :, None]
        * tx_weights[:, None, :]
    )

    # _assert_finite_local("beam_contributions", beam_contributions)

    H = beam_contributions.sum(dim=0)
    # _assert_finite_local("H", H)

    # A simple per_Gaussian usefulness score for prune/densify
    per_gaussian_importance = beam_contributions.abs().sum(dim=(1,2))
    # _assert_finite_local("per_gaussian_importance", per_gaussian_importance)

    return {
        "render": H,                     # now H itself is the predicted magnitude map
        "magnitude": H,
        "rx_weights": rx_weights,
        "tx_weights": tx_weights,
        "per_gaussian_importance": per_gaussian_importance,
        "beam_contributions": beam_contributions,
        "gain_weight": gain_weight,
    }