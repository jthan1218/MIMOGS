import math
import os

import numpy as np
import scipy.io as sio
import torch
from torch import nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

from utils.general_utils import (
    inverse_sigmoid,
    inverse_softplus,
    get_expon_lr_func,
    build_covariance_from_scaling_rotation,
    mkdir_p,
)

from typing import Optional, Dict, Any

class FourierFeatures(nn.Module):
    def __init__(self, in_dim=3, num_frequencies=6, include_input=True):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.out_dim = in_dim * ((1 if include_input else 0) + 2 * num_frequencies)

        freq_bands = (2.0 ** torch.arange(num_frequencies, dtype=torch.float32)) * math.pi
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_dim)
        [x, sin(f0*x), cos(f0*x), sin(f1*x), cos(f1*x), ...]
        """
        if self.num_frequencies == 0:
            if self.include_input:
                return x
            return x.new_empty(*x.shape[:-1], 0)

        freq_bands = self.freq_bands.to(device=x.device, dtype=x.dtype)

        # (..., F, D)
        x_proj = x.unsqueeze(-2) * freq_bands.view(*([1] * (x.dim() - 1)), -1, 1)

        sin_part = torch.sin(x_proj)
        cos_part = torch.cos(x_proj)

        # (..., F, 2, D) -> (..., 2*F*D)
        fourier = torch.stack((sin_part, cos_part), dim=-2).reshape(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat([x, fourier], dim=-1)
        return fourier

class DynamicGainNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        init_gain: float = 0.1,
        num_frequencies: int = 6,
        include_input: bool = True,
    ):
        super().__init__()

        self.pe = FourierFeatures(
            in_dim=3,
            num_frequencies=num_frequencies,
            include_input=include_input,
        )
        
        pe_dim = self.pe.out_dim
        # no-log-distance ablation: log-distance feature removed, so the MLP
        # input is [PE(xyz), PE(rx), PE(rel)] with no scalar log-distance.
        mlp_in_dim = pe_dim * 3

        self.net = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.zeros_(self.net[-1].weight)
        init_bias = float(inverse_softplus(torch.tensor(init_gain)))
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, 9)
        xyz = x[:, 0:3]
        rx  = x[:, 3:6]
        rel = x[:, 6:9]

        feat = torch.cat(
            [
                self.pe(xyz),
                self.pe(rx),
                self.pe(rel),
            ],
            dim=-1,
        )
        return self.net(feat)

    def _encoded_parts(self, x: torch.Tensor):
        """Return PE and reusable sine/cosine components.

        The returned encoding is exactly identical to ``FourierFeatures.forward``.
        Exposing the components lets ``forward_batched`` construct PE(xyz-rx)
        via angle-difference identities instead of evaluating trigonometric
        functions for every Gaussian/query pair.
        """
        if self.pe.num_frequencies == 0:
            encoded = x if self.pe.include_input else x.new_empty(*x.shape[:-1], 0)
            empty = x.new_empty(*x.shape[:-1], 0, x.shape[-1])
            return encoded, empty, empty

        freq = self.pe.freq_bands.to(device=x.device, dtype=x.dtype)
        proj = x.unsqueeze(-2) * freq.view(*([1] * (x.dim() - 1)), -1, 1)
        sin_part = torch.sin(proj)
        cos_part = torch.cos(proj)
        fourier = torch.stack((sin_part, cos_part), dim=-2).reshape(*x.shape[:-1], -1)
        encoded = torch.cat([x, fourier], dim=-1) if self.pe.include_input else fourier
        return encoded, sin_part, cos_part

    def forward_batched(self, xyz: torch.Tensor, rx: torch.Tensor) -> torch.Tensor:
        """Exact batched evaluation for N primitives and B UE positions.

        Args:
            xyz: ``(N,3)`` receive-side Gaussian anchors.
            rx: ``(B,3)`` UE positions.

        Returns:
            Raw dynamic gains with shape ``(B,N,1)``.

        This method is mathematically equivalent to applying ``forward`` to
        every concatenated ``[xyz, rx, xyz-rx]`` vector.  It is faster because
        the first linear layer is factorized into its three feature blocks and
        the position-only block is evaluated once per batch.
        """
        if xyz.dim() != 2 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must be (N,3), got {tuple(xyz.shape)}")
        if rx.dim() != 2 or rx.shape[-1] != 3:
            raise ValueError(f"rx must be (B,3), got {tuple(rx.shape)}")

        pe_xyz, sin_xyz, cos_xyz = self._encoded_parts(xyz)
        pe_rx, sin_rx, cos_rx = self._encoded_parts(rx)

        rel_raw = xyz.unsqueeze(0) - rx.unsqueeze(1)
        if self.pe.num_frequencies > 0:
            sin_rel = (
                sin_xyz.unsqueeze(0) * cos_rx.unsqueeze(1)
                - cos_xyz.unsqueeze(0) * sin_rx.unsqueeze(1)
            )
            cos_rel = (
                cos_xyz.unsqueeze(0) * cos_rx.unsqueeze(1)
                + sin_xyz.unsqueeze(0) * sin_rx.unsqueeze(1)
            )
            rel_fourier = torch.stack((sin_rel, cos_rel), dim=-2).reshape(
                rel_raw.shape[0], rel_raw.shape[1], -1
            )
            pe_rel = (
                torch.cat([rel_raw, rel_fourier], dim=-1)
                if self.pe.include_input
                else rel_fourier
            )
        else:
            pe_rel = rel_raw if self.pe.include_input else rel_raw.new_empty(
                rel_raw.shape[0], rel_raw.shape[1], 0
            )

        first = self.net[0]
        pe_dim = self.pe.out_dim
        w_xyz = first.weight[:, :pe_dim]
        w_rx = first.weight[:, pe_dim : 2 * pe_dim]
        w_rel = first.weight[:, 2 * pe_dim :]

        # Bias is added only in the relative block; the three terms are then
        # broadcast and summed to reproduce Linear([PE_xyz, PE_rx, PE_rel]).
        h_xyz = F.linear(pe_xyz, w_xyz, None)                       # (N,H)
        h_rx = F.linear(pe_rx, w_rx, None)                          # (B,H)
        h_rel = F.linear(pe_rel.reshape(-1, pe_dim), w_rel, first.bias)
        h = h_rel.view(rx.shape[0], xyz.shape[0], -1)
        h = h + h_xyz.unsqueeze(0) + h_rx.unsqueeze(1)

        h = self.net[1](h)
        h = self.net[2](h)
        h = self.net[3](h)
        h = self.net[4](h)
        return h

class GaussianModel:
    """MIMOGS Gaussian scene model

    Learnable attributes per Gaussian:
    - mean                  : xyz      (rx side) + xyz_tx (tx side)
    - covariance            : rotation + scaling      (rx side)
                              rotation_tx + scaling_tx (tx side)
    - opacity-like weight   :opacity

    A primitive is therefore a *pair* of 3D Gaussians -- one seen from the UE,
    one seen from the BS -- tied together by the single shared per-primitive
    gain.  Set ``tie_covariance=True`` to force the two ends to share one 3D
    shape, which reproduces the earlier single-covariance model exactly.
    """

    def __init__(
        self,
        target_gaussians: int = 50_000,
        optimizer_type: str = "default",
        device: str = "cuda",
        init_range: float = 5.0,
        tie_covariance: bool = False,
        gain_pe_frequencies: int = 6,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.optimizer_type = optimizer_type
        self.target_gaussians = target_gaussians
        self.init_range = init_range
        self.tie_covariance = bool(int(tie_covariance))
        # Fourier bands for the dynamic-gain MLP's positional encoding. 6 is
        # the historical default; 0 hands the raw coordinates to the MLP.
        self.gain_pe_frequencies = int(gain_pe_frequencies)

        self._xyz = torch.empty(0, device = self.device)
        self._xyz_tx = torch.empty(0, device = self.device)
        self._scaling = torch.empty(0, device = self.device)
        self._rotation = torch.empty(0, device = self.device)
        self._scaling_tx = torch.empty(0, device = self.device)
        self._rotation_tx = torch.empty(0, device = self.device)
        self._opacity = torch.empty(0, device = self.device)
        # self._gain_mag = torch.empty(0, device = self.device)

        self.optimizer = None
        self.xyz_scheduler_args = None

        self.xyz_gradient_accum = torch.empty(0,device = self.device)
        self.grad_denom = torch.empty(0,device = self.device)
        self.importance_accum = torch.empty(0,device = self.device)
        self.importance_denom = torch.empty(0,device = self.device)
        
        self.dynamic_gain_net = DynamicGainNet(
            num_frequencies=self.gain_pe_frequencies
        ).to(self.device)
        self.dynamic_gain_optimizer = None
        self.dynamic_gain_scheduler_args = None
        
        self.setup_functions()

    def setup_functions(self):
        self.scaling_activation = lambda x: torch.exp(torch.clamp(x, min=-10.0, max=5.0))
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = lambda x: F.normalize(x, dim=-1)

        # self.gain_mag_activation = F.softplus
        # self.gain_mag_inverse_activation = inverse_softplus

        self.covariance_activation = build_covariance_from_scaling_rotation


    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_xyz_tx(self):
        return self._xyz_tx

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_scaling_tx(self):
        if self.tie_covariance:
            return self.get_scaling
        return self.scaling_activation(self._scaling_tx)

    @property
    def get_rotation_tx(self):
        if self.tie_covariance:
            return self.get_rotation
        return self.rotation_activation(self._rotation_tx)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    # @property
    # def get_gain_mag(self):
    #     return self.gain_mag_activation(self._gain_mag)

    # @property
    # def get_gain_weight(self):
    #     return self.get_opacity * self.get_gain_mag

    def get_covariance(self, scaling_modifier: float = 1.0):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self.get_rotation, return_strip = False
        )

    def get_covariance_tx(self, scaling_modifier: float = 1.0):
        """Tx-side 3D covariance of every primitive.

        Returns the Rx-side covariance when ``tie_covariance`` is set, so the
        two projections then share one shape exactly as before.
        """
        if self.tie_covariance:
            return self.get_covariance(scaling_modifier)
        return self.covariance_activation(
            self.get_scaling_tx, scaling_modifier, self.get_rotation_tx, return_strip = False
        )

    def _sync_tied_covariance(self):
        """Alias the Tx-side covariance parameters onto the Rx-side ones.

        Only used when ``tie_covariance`` is set.  The Tx parameters are then
        not optimized (and not carried through the densification bookkeeping),
        so aliasing keeps ``_scaling_tx`` / ``_rotation_tx`` consistent in both
        value and shape with what the model actually renders.
        """
        if not self.tie_covariance:
            return
        self._scaling_tx = self._scaling
        self._rotation_tx = self._rotation

    # ------------------------------------------------------------------
    # Init / save / restore
    # ------------------------------------------------------------------
    def _build_initial_points(self, vertices_path: Optional[str] = None) -> torch.Tensor:
        fused_point_cloud = None

        if vertices_path is not None and os.path.exists(vertices_path):
            try:
                mat = sio.loadmat(vertices_path)
                vertices = mat.get("vertices", None)
                if vertices is not None and vertices.size > 0:
                    base_points = torch.tensor(
                        vertices, dtype = torch.float32, device = self.device
                    )
                    base_count = base_points.shape[0]

                    if base_count > self.target_gaussians:
                        fused_point_cloud = base_points[: self.target_gaussians]
                    else:
                        repeat_idx = torch.randint(
                            0,
                            base_count,
                            (self.target_gaussians,),
                            device = self.device,
                        )
                        jitter = (
                            torch.randn((self.target_gaussians, 3), device = self.device) * 0.01
                        )
                        fused_point_cloud = base_points[repeat_idx] + jitter
            except Exception as exc:
                print(
                    f"Failed to load vertices from {vertices_path}."
                    f"Fallback to random initialization: {exc}"
                )

        if fused_point_cloud is None:
            fused_point_cloud = (
                torch.randn((self.target_gaussians, 3), device = self.device).float() * self.init_range
            )

        return fused_point_cloud
    

    def gaussian_init(self, vertices_path: Optional[str] = None):
        fused_point_cloud = self._build_initial_points(vertices_path = vertices_path)
        n_points = fused_point_cloud.shape[0]

        scene_scale = fused_point_cloud.std(dim = 0).mean().clamp(min = 1e-3)
        init_scale = torch.full(
            (n_points, 3),
            0.5 * scene_scale.item(),
            dtype = torch.float32,
            device = self.device,
        )
        scales_raw = self.scaling_inverse_activation(init_scale)

        rots = torch.zeros((n_points, 4), dtype = torch.float32, device = self.device)
        rots[:, 0] = 1.0

        opacities_raw = self.inverse_opacity_activation(
            0.1 * torch.ones((n_points, 1), dtype=torch.float32, device = self.device)
        )

        # gain_mag_raw = self.gain_mag_inverse_activation(
        #     0.1 * torch.ones((n_points, 1), dtype = torch.float32, device = self.device)
        # )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._xyz_tx = nn.Parameter(fused_point_cloud.clone().requires_grad_(True))
        self._scaling = nn.Parameter(scales_raw.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        # The Tx side starts as an exact clone of the Rx side, so training starts
        # from the tied behaviour and diverges only through the gradients each
        # side receives from its own projection.
        self._scaling_tx = nn.Parameter(scales_raw.detach().clone().requires_grad_(True))
        self._rotation_tx = nn.Parameter(rots.detach().clone().requires_grad_(True))
        self._opacity = nn.Parameter(opacities_raw.requires_grad_(True))
        # self._gain_mag = nn.Parameter(gain_mag_raw.requires_grad_(True))

        self._sync_tied_covariance()
        self._reset_statistics()
        print(f"[GaussianModel] Number of points at initialization: {n_points}")

    def capture(self):
        return (
            self.target_gaussians,
            self.optimizer_type,
            self.init_range,
            self._xyz.detach(),
            self._scaling.detach(),
            self._rotation.detach(),
            self._opacity.detach(),
            # self._gain_mag.detach(),
            self.xyz_gradient_accum.detach(),
            self.grad_denom.detach(),
            self.importance_accum.detach(),
            self.importance_denom.detach(),
            None if self.optimizer is None else self.optimizer.state_dict(),

            self.dynamic_gain_net.state_dict(),
            None if self.dynamic_gain_optimizer is None else self.dynamic_gain_optimizer.state_dict(),
            # Appended at the end so legacy positional consumers (indices 0-2, 11, 13)
            # keep working. Old checkpoints without this entry restore via fallback below.
            self._xyz_tx.detach(),
            # Same pattern for the decoupled Tx-side covariance. When tied these
            # alias the Rx-side tensors (see _sync_tied_covariance), so a tied run
            # reloads into an untied model with identical rendering behaviour.
            self._scaling_tx.detach(),
            self._rotation_tx.detach(),
        )

    def restore(self, model_args, training_args):
        model_args = tuple(model_args)
        (
            self.target_gaussians,
            self.optimizer_type,
            self.init_range,
            xyz,
            scaling,
            rotation,
            opacity,
            # gain_mag,
            xyz_gradient_accum,
            grad_denom,
            importance_accum,
            importance_denom,
            opt_dict,

            dynamic_gain_net_dict,
            dynamic_gain_opt_dict,
        ) = model_args[:14]

        # Backward-compat: pre-Tx-anchor checkpoints have len(model_args)==14,
        # pre-Tx-covariance checkpoints have len(model_args)==15. The trailing
        # slots are also where discontinued experiments parked their own extra
        # tensors, so an entry is only accepted when it is a tensor of the shape
        # this slot is supposed to hold; anything else falls back to the
        # receive-side clone.
        def _optional(index: int, like: torch.Tensor, label: str):
            if len(model_args) < index + 1:
                return None
            value = model_args[index]
            if value is None:
                return None
            if not torch.is_tensor(value) or tuple(value.shape) != tuple(like.shape):
                print(
                    f"[GaussianModel] Ignoring unexpected checkpoint entry at index "
                    f"{index} for {label} (expected tensor of shape {tuple(like.shape)})"
                )
                return None
            return value

        xyz_tx = _optional(14, xyz, "xyz_tx")
        scaling_tx = _optional(15, scaling, "scaling_tx")
        rotation_tx = _optional(16, rotation, "rotation_tx")

        self._xyz = nn.Parameter(xyz.to(self.device).requires_grad_(True))
        self._scaling = nn.Parameter(scaling.to(self.device).requires_grad_(True))
        self._rotation = nn.Parameter(rotation.to(self.device).requires_grad_(True))
        self._opacity = nn.Parameter(opacity.to(self.device).requires_grad_(True))
        # self._gain_mag = nn.Parameter(gain_mag.to(self.device).requires_grad_(True))

        if xyz_tx is not None:
            self._xyz_tx = nn.Parameter(xyz_tx.to(self.device).requires_grad_(True))
        else:
            self._xyz_tx = nn.Parameter(self._xyz.detach().clone().requires_grad_(True))

        # Missing Tx-side covariance (older checkpoint) falls back to a clone of
        # the Rx side, which is exactly the shared-covariance behaviour the
        # checkpoint was trained with.
        if scaling_tx is not None:
            self._scaling_tx = nn.Parameter(scaling_tx.to(self.device).requires_grad_(True))
        else:
            self._scaling_tx = nn.Parameter(self._scaling.detach().clone().requires_grad_(True))

        if rotation_tx is not None:
            self._rotation_tx = nn.Parameter(rotation_tx.to(self.device).requires_grad_(True))
        else:
            self._rotation_tx = nn.Parameter(self._rotation.detach().clone().requires_grad_(True))

        self._sync_tied_covariance()

        self.training_setup(training_args)

        self.xyz_gradient_accum = xyz_gradient_accum.to(self.device)
        self.grad_denom = grad_denom.to(self.device)
        self.importance_accum = importance_accum.to(self.device)
        self.importance_denom = importance_denom.to(self.device)
        
        if opt_dict is not None:
            self._load_optimizer_state(opt_dict)

        if dynamic_gain_net_dict is not None:
            self.dynamic_gain_net.load_state_dict(dynamic_gain_net_dict)

        if dynamic_gain_opt_dict is not None and self.dynamic_gain_optimizer is not None:
            self.dynamic_gain_optimizer.load_state_dict(dynamic_gain_opt_dict)

    def _load_optimizer_state(self, opt_dict: Dict[str, Any]):
        """Load Adam state, tolerating a different set of parameter groups.

        Checkpoints written before the Tx-side covariance (or before the Tx
        anchor) contain fewer groups than the current optimizer, and a tied
        model has fewer groups than an untied checkpoint.  Groups are therefore
        matched by name; anything unmatched simply starts from a fresh state.
        """
        current = self.optimizer.state_dict()
        saved_groups = opt_dict.get("param_groups", [])

        if len(saved_groups) == len(current["param_groups"]):
            self.optimizer.load_state_dict(opt_dict)
            return

        saved_state = opt_dict.get("state", {})
        saved_by_name = {}
        for saved_group in saved_groups:
            name = saved_group.get("name", None)
            params = saved_group.get("params", [])
            if name is None or len(params) != 1:
                continue
            saved_by_name[name] = (saved_group, saved_state.get(params[0], None))

        matched = []
        for group in current["param_groups"]:
            entry = saved_by_name.get(group.get("name", None), None)
            if entry is None:
                continue
            saved_group, param_state = entry
            for key, value in saved_group.items():
                if key not in ("params", "name"):
                    group[key] = value
            if param_state is not None:
                current["state"][group["params"][0]] = param_state
            matched.append(group["name"])

        self.optimizer.load_state_dict(current)
        print(
            "[GaussianModel] Optimizer state restored partially "
            f"({len(saved_groups)} saved groups -> {len(current['param_groups'])} current); "
            f"matched: {matched}"
        )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def _reset_statistics(self):
        n = self._xyz.shape[0]
        self.xyz_gradient_accum = torch.zeros((n, 1), device = self.device)
        self.grad_denom = torch.zeros((n, 1), device = self.device)
        self.importance_accum = torch.zeros((n, 1), device = self.device)
        self.importance_denom = torch.zeros((n, 1), device = self.device)

    def training_setup(self, training_args):
        self._reset_statistics()
        
        param_groups = [
            {"params": [self._xyz], "lr": training_args.position_lr_init, "name": "xyz"},
            {"params": [self._xyz_tx], "lr": training_args.position_lr_init, "name": "xyz_tx"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
            # {"params": [self._gain_mag], "lr": training_args.gain_lr, "name": "gain_mag"},
        ]

        # The Tx-side covariance reuses the Rx-side learning rates. When tied it
        # is not optimized at all, so the optimizer is identical to before.
        if not self.tie_covariance:
            param_groups += [
                {"params": [self._scaling_tx], "lr": training_args.scaling_lr, "name": "scaling_tx"},
                {"params": [self._rotation_tx], "lr": training_args.rotation_lr, "name": "rotation_tx"},
            ]

        if getattr(training_args, "optimizer_type", self.optimizer_type) == "adamw":
            self.optimizer = torch.optim.AdamW(param_groups, lr = 0.0, eps = 1e-8)
        else:
            self.optimizer = torch.optim.Adam(param_groups, lr = 0.0, eps = 1e-8)

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init = training_args.position_lr_init,
            lr_final = training_args.position_lr_final,
            lr_delay_mult = training_args.position_lr_delay_mult,
            max_steps = training_args.position_lr_max_steps,
        )

        self.opacity_scheduler_args = get_expon_lr_func(
            lr_init=training_args.opacity_lr,
            lr_final=training_args.opacity_lr_final,
            lr_delay_mult=1.0,
            max_steps=training_args.iterations,
        )

        # self.gain_scheduler_args = get_expon_lr_func(
        #     lr_init=training_args.gain_lr,
        #     lr_final=training_args.gain_lr_final,
        #     lr_delay_mult=1.0,
        #     max_steps=training_args.iterations,
        # )

        self.dynamic_gain_optimizer = torch.optim.Adam(
            self.dynamic_gain_net.parameters(),
            lr=training_args.dynamic_gain_lr,
            eps=1e-8,
        )

        self.dynamic_gain_scheduler_args = get_expon_lr_func(
            lr_init=training_args.dynamic_gain_lr,
            lr_final=training_args.dynamic_gain_lr_final,
            lr_delay_mult=1.0,
            max_steps=training_args.iterations,
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "xyz_tx":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "opacity":
                lr = self.opacity_scheduler_args(iteration)
                param_group["lr"] = lr
            # elif param_group["name"] == "gain_mag":
            #     lr = self.gain_scheduler_args(iteration)
            #     param_group["lr"] = lr

        if self.dynamic_gain_optimizer is not None:
            dyn_lr = self.dynamic_gain_scheduler_args(iteration)
            for param_group in self.dynamic_gain_optimizer.param_groups:
                param_group["lr"] = dyn_lr

    def get_dynamic_gain_weight_batched(self, rx_pos: torch.Tensor) -> torch.Tensor:
        """Return location-conditioned primitive gains for B UE positions.

        Args:
            rx_pos: ``(B,3)`` or ``(3,)``.
        Returns:
            ``(B,N)`` tensor containing ``rho_k d_k(p)``.
        """
        rx = rx_pos.to(self.device, dtype=self.get_xyz.dtype)
        if rx.dim() == 1:
            rx = rx.view(1, 3)
        if rx.dim() != 2 or rx.shape[-1] != 3:
            raise ValueError(f"rx_pos must be (3,) or (B,3), got {tuple(rx.shape)}")

        raw_gain = self.dynamic_gain_net.forward_batched(self.get_xyz, rx)
        dynamic_gain = F.softplus(raw_gain).squeeze(-1)            # (B,N)
        return dynamic_gain * self.get_opacity.squeeze(-1).unsqueeze(0)

    def get_dynamic_gain_weight(self, rx_pos: torch.Tensor) -> torch.Tensor:
        """Backward-compatible single-query wrapper returning ``(N,1)``."""
        return self.get_dynamic_gain_weight_batched(rx_pos).squeeze(0).unsqueeze(-1)

    # ------------------------------------------------------------------
    # Statistics for pruning / densification
    # ------------------------------------------------------------------

    def accumulate_training_stats(self, importance: Optional[torch.Tensor] = None):
        if self._xyz.grad is None:
            return

        xyz_grad = torch.norm(self._xyz.grad.detach(), dim=-1, keepdim=True)
        self.xyz_gradient_accum += xyz_grad
        self.grad_denom += 1.0

        if importance is not None:
            if importance.dim() == 1:
                importance = importance.unsqueeze(-1)
            self.importance_accum += importance.detach().to(self.device)
            self.importance_denom += 1.0

    def get_avg_xyz_grad(self):
        denom = torch.clamp(self.grad_denom, min=1.0)
        return self.xyz_gradient_accum / denom

    def get_avg_importance(self):
        denom = torch.clamp(self.importance_denom, min=1.0)
        return self.importance_accum / denom

    # ------------------------------------------------------------------
    # PLY I/O
    # ------------------------------------------------------------------
    def construct_list_of_attributes(self):
        attrs = ["x", "y", "z", "nx", "ny", "nz", "opacity"]
        attrs += [f"scale_{i}" for i in range(3)]
        attrs += [f"rot_{i}" for i in range(4)]
        # attrs += ["gain_mag"]
        return attrs

    def save_ply(self, path: str):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        opacities = self.get_opacity.detach().cpu().numpy()
        scales = self.get_scaling.detach().cpu().numpy()
        rotations = self.get_rotation.detach().cpu().numpy()
        # gain_mag = self.get_gain_mag.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            [xyz, normals, opacities, scales, rotations], axis = 1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path: str):
        plydata = PlyData.read(path)

        xyz = np.stack(
            [
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ],
            axis = 1,
        )

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # gain_mag = np.asarray(plydata.elements[0]["gain_mag"])[..., np.newaxis]

        xyz_t = torch.tensor(xyz, dtype=torch.float32, device=self.device)
        opacity_t = torch.tensor(opacities, dtype=torch.float32, device=self.device)
        scale_t = torch.tensor(scales, dtype=torch.float32, device=self.device)
        rot_t = torch.tensor(rots, dtype=torch.float32, device=self.device)
        # gain_mag_t = torch.tensor(gain_mag, dtype=torch.float32, device=self.device)

        self._xyz = nn.Parameter(xyz_t.requires_grad_(True))
        # PLY format carries only the rx anchor; tie the tx anchor to it on reload.
        self._xyz_tx = nn.Parameter(xyz_t.clone().requires_grad_(True))
        self._opacity = nn.Parameter(
            self.inverse_opacity_activation(opacity_t).requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            self.scaling_inverse_activation(torch.clamp(scale_t, min=1e-8)).requires_grad_(True)
        )
        self._rotation = nn.Parameter(rot_t.requires_grad_(True))
        # PLY format carries only the rx covariance; tie the tx side to it on reload.
        self._scaling_tx = nn.Parameter(self._scaling.detach().clone().requires_grad_(True))
        self._rotation_tx = nn.Parameter(rot_t.clone().requires_grad_(True))
        # self._gain_mag = nn.Parameter(
        #     self.gain_mag_inverse_activation(torch.clamp(gain_mag_t, min=1e-8)).requires_grad_(True)
        # )

        self._sync_tied_covariance()
        self._reset_statistics()

    # ------------------------------------------------------------------
    # Optimizer-safe tensor replacement helpers
    # ------------------------------------------------------------------

    def replace_tensor_to_optimizer(self, tensor: torch.Tensor, name:str):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] != name:
                continue

            stored_state = self.optimizer.state.get(group["params"][0], None)

            if stored_state is not None:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

            else:
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))

            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask: torch.Tensor):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            old_param = group["params"][0]
            stored_state = self.optimizer.state.get(old_param, None)

            new_tensor = old_param[mask]

            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[old_param]
                group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))

            optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict: Dict[str, torch.Tensor]):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            extension_tensor = tensors_dict[group["name"]]
            old_param = group["params"][0]
            stored_state = self.optimizer.state.get(old_param, None)

            new_tensor = torch.cat((old_param, extension_tensor), dim=0)

            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )

                del self.optimizer.state[old_param]
                group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))

            optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # ------------------------------------------------------------------
    # Prune / densify
    # ------------------------------------------------------------------

    def prune_points(self, mask: torch.Tensor):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._xyz_tx = optimizable_tensors["xyz_tx"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # self._gain_mag = optimizable_tensors["gain_mag"]

        if self.tie_covariance:
            self._sync_tied_covariance()
        else:
            self._scaling_tx = optimizable_tensors["scaling_tx"]
            self._rotation_tx = optimizable_tensors["rotation_tx"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.grad_denom = self.grad_denom[valid_points_mask]
        self.importance_accum = self.importance_accum[valid_points_mask]
        self.importance_denom = self.importance_denom[valid_points_mask]

    def densification_postfix(
        self,
        new_xyz: torch.Tensor,
        new_opacity: torch.Tensor,
        new_scaling: torch.Tensor,
        new_rotation: torch.Tensor,
        new_xyz_tx: torch.Tensor,
        new_scaling_tx: Optional[torch.Tensor] = None,
        new_rotation_tx: Optional[torch.Tensor] = None,
        # new_gain_mag: torch.Tensor,
    ):
        tensors_dict = {
            "xyz": new_xyz,
            "xyz_tx": new_xyz_tx,
            "opacity": new_opacity,
            "scaling": new_scaling,
            "rotation": new_rotation,
            # "gain_mag": new_gain_mag,
        }

        if not self.tie_covariance:
            # Callers that predate the decoupled covariance simply duplicate the
            # rx-side shape onto the new tx-side entries.
            tensors_dict["scaling_tx"] = (
                new_scaling.clone() if new_scaling_tx is None else new_scaling_tx
            )
            tensors_dict["rotation_tx"] = (
                new_rotation.clone() if new_rotation_tx is None else new_rotation_tx
            )

        optimizable_tensors = self.cat_tensors_to_optimizer(tensors_dict)

        self._xyz = optimizable_tensors["xyz"]
        self._xyz_tx = optimizable_tensors["xyz_tx"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # self._gain_mag = optimizable_tensors["gain_mag"]

        if self.tie_covariance:
            self._sync_tied_covariance()
        else:
            self._scaling_tx = optimizable_tensors["scaling_tx"]
            self._rotation_tx = optimizable_tensors["rotation_tx"]

        self._reset_statistics()

    def get_pair_scaling(self):
        """Elementwise max of the rx- and tx-side scalings.

        Densification and pruning decisions act on the primitive as a whole, so
        the pair is described by the larger of the two extents on every axis.
        Reduces to ``get_scaling`` when the covariance is tied.
        """
        return torch.maximum(self.get_scaling, self.get_scaling_tx)

    def densify_and_clone(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        clone_scale_threshold: float,
        importance_threshold: float = 0.0,
    ):
        avg_importance = self.get_avg_importance().squeeze(-1)
        selected_pts_mask = (grads.squeeze(-1) >= grad_threshold)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            self.get_pair_scaling().max(dim=1).values <= clone_scale_threshold,
        )

        if importance_threshold > 0.0:
            selected_pts_mask = torch.logical_and(
                selected_pts_mask, avg_importance >= importance_threshold
            )

        if selected_pts_mask.sum() == 0:
            return

        new_xyz = self._xyz[selected_pts_mask].clone()
        new_xyz_tx = self._xyz_tx[selected_pts_mask].clone()
        new_opacity = self._opacity[selected_pts_mask].clone()
        new_scaling = self._scaling[selected_pts_mask].clone()
        new_rotation = self._rotation[selected_pts_mask].clone()
        # Both ends of the primitive are cloned together; the child is a full copy
        # of the pair, never a mix of two different primitives.
        new_scaling_tx = self._scaling_tx[selected_pts_mask].clone()
        new_rotation_tx = self._rotation_tx[selected_pts_mask].clone()
        # new_gain_mag = self._gain_mag[selected_pts_mask].clone()


        self.densification_postfix(
            new_xyz, new_opacity, new_scaling, new_rotation, new_xyz_tx,
            new_scaling_tx, new_rotation_tx,
        )

    def densify_and_split(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        split_scale_threshold: float,
        importance_threshold: float = 0.0,
        n_splits: int = 2,
    ):
        avg_importance = self.get_avg_importance().squeeze(-1)
        selected_pts_mask = (grads.squeeze(-1) >= grad_threshold)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            self.get_pair_scaling().max(dim=1).values > split_scale_threshold,
        )
        if importance_threshold > 0.0:
            selected_pts_mask = torch.logical_and(
                selected_pts_mask, avg_importance >= importance_threshold
            )

        n_selected = int(selected_pts_mask.sum().item())
        if n_selected == 0:
            return

        stds = self.get_scaling[selected_pts_mask].repeat(n_splits, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)

        rots = self.get_rotation[selected_pts_mask].repeat(n_splits,1)
        from utils.general_utils import build_rotation  # local import to avoid clutter
        rot_mats = build_rotation(rots)

        base_xyz = self.get_xyz[selected_pts_mask].repeat(n_splits, 1)
        offsets = torch.bmm(rot_mats, samples.unsqueeze(-1)).squeeze(-1)
        new_xyz = offsets + base_xyz

        new_scaling = self.scaling_inverse_activation(
            torch.clamp(
                self.get_scaling[selected_pts_mask].repeat(n_splits, 1) / (0.8 * n_splits),
                min=1e-8,
            )
        )
        new_rotation = self.get_rotation[selected_pts_mask].repeat(n_splits, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(n_splits, 1)
        # new_gain_mag = self._gain_mag[selected_pts_mask].repeat(n_splits, 1)

        base_xyz_tx = self._xyz_tx[selected_pts_mask].repeat(n_splits, 1)

        if self.tie_covariance:
            # Mirror the per-child offset on the Tx anchor so the (rx, tx)
            # pair-relationship learned so far is preserved by each child. No
            # extra random numbers are drawn, so the tied path is bit-identical
            # to the shared-covariance implementation.
            new_xyz_tx = offsets + base_xyz_tx
            new_scaling_tx = new_scaling
            new_rotation_tx = new_rotation
        else:
            # The Tx anchor has its own covariance, so its child offsets are
            # drawn from that covariance instead of being copied from the Rx
            # side. Both children still belong to the same primitive pair.
            stds_tx = self.get_scaling_tx[selected_pts_mask].repeat(n_splits, 1)
            samples_tx = torch.normal(mean=means, std=stds_tx)
            rots_tx = self.get_rotation_tx[selected_pts_mask].repeat(n_splits, 1)
            rot_mats_tx = build_rotation(rots_tx)
            offsets_tx = torch.bmm(rot_mats_tx, samples_tx.unsqueeze(-1)).squeeze(-1)
            new_xyz_tx = offsets_tx + base_xyz_tx

            new_scaling_tx = self.scaling_inverse_activation(
                torch.clamp(
                    self.get_scaling_tx[selected_pts_mask].repeat(n_splits, 1) / (0.8 * n_splits),
                    min=1e-8,
                )
            )
            new_rotation_tx = self.get_rotation_tx[selected_pts_mask].repeat(n_splits, 1)

        self.densification_postfix(
            new_xyz, new_opacity, new_scaling, new_rotation, new_xyz_tx,
            new_scaling_tx, new_rotation_tx,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(n_splits * n_selected, device = self.device, dtype = torch.bool)
            )
        )
        self.prune_points(prune_filter)

    def densify_and_prune(
        self,
        max_grad: float,
        min_opacity: float,
        # min_gain_mag: float,
        clone_scale_threshold: float,
        split_scale_threshold: float,
        importance_threshold: float = 0.0,
        max_scale: Optional[float] = None,
        n_splits: int = 2,
    ):
        grads = self.get_avg_xyz_grad()
        grads[torch.isnan(grads)] = 0.0

        self.densify_and_clone(
            grads=grads,
            grad_threshold = max_grad,
            clone_scale_threshold = clone_scale_threshold,
            importance_threshold = importance_threshold,
        )

        self.densify_and_split(
            grads = grads,
            grad_threshold = max_grad,
            split_scale_threshold = split_scale_threshold,
            importance_threshold = importance_threshold,
            n_splits = n_splits,
        )

        prune_mask = (self.get_opacity.squeeze(-1) < min_opacity)
        # prune_mask = torch.logical_or(
        #     prune_mask, self.get_gain_mag.squeeze(-1) < min_gain_mag
        # )

        if max_scale is not None:
            prune_mask = torch.logical_or(
                prune_mask, self.get_pair_scaling().max(dim=1).values > max_scale
            )

        self.prune_points(prune_mask)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset_opacity(self, max_opacity: float = 0.01):
        new_opacity = self.inverse_opacity_activation(
            torch.minimum(
                self.get_opacity,
                torch.full_like(self.get_opacity, max_opacity)
            )
        )
        optimizable = self.replace_tensor_to_optimizer(new_opacity, "opacity")
        self._opacity = optimizable["opacity"]