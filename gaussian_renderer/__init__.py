import math
from symbol import break_stmt
from typing import Dict, Tuple, Optional

import torch

from scene.gaussian_model import GaussianModel

def _ensure_pos_shape(x: torch.Tensor) -> torch.Tensor:
    """Accepts shape (3,) or (1,3), returns shape (3,)"""

    if x.ndim() == 2 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.dim() != 1 or x.shape[0] != 3:
        raise ValueError(f"Position must have shape (3,) or (1,3), got {tuple(x.shape)}")
    return x

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

    unit_dir, dist = _direction_and_distance(means, array_pos)      # (N,3), (N,1)
    uv_mean = _uv_from_unit_direction(unit_dir)                     # (N,2)

    # Jacobian of normalized vector: J = (I - uu^T) / ||r||
    eye3 = torch.eye(3, device = device, dtype = dtype).unsqueeze(0).expand(means.shape[0], -1, -1)
    uuT = unit_dir.unsqueeze(-1) @ unit_dir.unsqueeze(-2)           # (N,3,3)
    J_unit = (eye3 - uuT) / dist.unsqueeze(-1)                      # (N,3,3)

    # uv = [unit_dir_y, unit_dir_z], so keep rows 1 and 2
    J_uv = J_unit[:, 1:3, :]                                       # (N,2,3)

    cov_uv = J_uv @ covariances @ J_uv.transpose(-1, -2)           # (N,2,2)

    eye2 = torch.eye(2, device=device, dtype=dtype).unsqueeze(0).expand(means.shape[0], -1, -1)
    cov_uv = cov_uv + covariance_floor * eye2
    
    return uv_mean, cov_uv, dist


def _gaussian_beam_weights_from_uv(
    uv_mean: torch.Tensor,          # (N,2)
    cov_uv: torch.Tensor,           # (N,2,2)
    beam_centers_uv: torch.Tensor,  # (B,2)
    normalize: bool = True,
    weight_floor: float = 1e-12,
) -> torch.Tensor:
    f"""
    Soft projection of 3D Gaussian onto beam-domiain grid.

    weight_n[b] = exp(-0.5 * (c-b - mu_n)^T Sigma_n^{-1} (c-b - mu_n))

    If normalize = True, weights are normalized to sum to 1 across beams for each Gaussian.
    This lets covariance control spread, while gain/opacity controls total magnitude.m
    """

    delta = beam_centers_uv.unsqueeze(0) - uv_mean.unsqueeze(1)     # (N,B,2)
    inv_cov_uv = torch.linalg.inv(cov_uv)                           # (N,2,2)

    mahal = torch.einsum("nbi,nij,nbj->nb", delta, inv_cov_uv, delta) # (N,B)
    weights = torch.exp(-0.5 * mahal)

    weights = torch.where(weights < weight_floor, torch.zeros_like(weights), weights)

    if normalize:
        denom = weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        weights = weights / denom

    return weights

def _apply_geometric_phase(
    complex_weight: torch.Tensor, # (N,1) complex
    means: torch.Tensor,          # (N,3)
    tx_pos: torch.Tensor,         # (3,)
    rx_pos: torch.Tensor,
    carrier_frequency_hz: float,
) -> torch.Tensor:
    """
    Optional bistatic propagation phase:
        exp(-j 2pi / lambda * (||mu - tx|| + ||mu - rx||))
    """
    c0 = 299_792_458.0 # m/s
    wavelength = c0 / carrier_frequency_hz

    d_tx = torch.norm(means - tx_pos.unsqueeze(0), dim=-1, keepdim=True)
    d_rx = torch.norm(means = rx_pos.unsqueeze(0), dim=-1, keepdim=True)
    path_length = d_tx + d_rx

    phase = -2.0 * math.pi * path_length / wavelength
    geom = torch.exp(1j * phase)

    return complex_weight * geom

def render(
    rx_pos: torch.Tensor,
    tx_pos: torch.Tensors,
    pc: GaussianModel,
    rx_shape: Tuple[int, int] = (2, 2),     # (horizontal, vertical)
    tx_shape: Tuple[int, int] = (4, 4),     # (horizontal, vertical)
    scaling_modifier: float = 1.0,
    use_geometric_phase: bool = True,
    carrier_frequency_hz: float = 0.0,
    normalize_beam_weights: bool = True,
    covariance_floor: float = 1e-6,
    weight_floor: float = 1e-12,
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

    means = pc.get_xyz      # (N,3)
    covariances = pc.get_covariance(scaling_modifier) # (N,3,3)
    complex_weight = pc.get_complex_weight         # (N,1) complex

    if use_geometric_phase:
        if carrier_frequency_hz <= 0.0:
            raise ValueError("Carrier frequency must be positive when use_geometric_phase=True")
        complex_weight = _apply_geometric_phase(
            complex_weight = complex_weight,
            means = means,
            tx_pos = tx_pos,
            rx_pos = rx_pos,
            carrier_frequency_hz = carrier_frequency_hz,
        )

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
    rx_weights = _gaussian_beam_weights_from_uv(
        uv_mean = rx_uv_mean,
        cov_uv = rx_cov_uv,
        beam_centers_uv = rx_beam_centers_uv,
        normalize = normalize_beam_weights,
        weight_floor = weight_floor,
    )

    # ------------------------------------------------------------------
    # Covariance-aware soft projection to Tx beam-domain
    # ------------------------------------------------------------------
    tx_uv_mean, tx_cov_uv, _ = _projected_angular_covariance(
        means=means,
        covariances=covariances,
        array_pos = tx_pos,
        covariance_floor = covariance_floor,
    )

    tx_weights = _gaussian_beam_weights_from_uv(
        uv_mean = tx_uv_mean,
        cov_uv = tx_cov_uv,
        beam_centers_uv = tx_beam_centers_uv,
        normalize = normalize_beam_weights,
        weight_floor = weight_floor,
    )

    # ------------------------------------------------------------------
    # Beamspace splatting / superposition
    # H_n[p,q] = c_n * r_n[p] * t_n[q]
    # ------------------------------------------------------------------ 
    beam_contributions = (
        complex_weight[:, None, None]
        * rx_weights[:, :, None].to(complex_weight.dtype)
        * tx_weights[:, None, :].to(complex_weight.dtype)
    ) # (N, Nr, Nt)

    H = beam_contributions.sum(dim=0) # (Nr, Nt)

    # A simple per_Gaussian usefulness score for prune/densify
    per_gaussian_importance = beam_contributions.abs().sum(dim=(1,2))

    return {
        "render": H,
        "magnitude": torch.abs(H),
        "phase": torch.angle(H),
        "rx_weights": rx_weights,
        "tx_weights": tx_weights,
        "per_Gaussian_importance": per_gaussian_importance,
        "beam_contributions": beam_contributions,
    }