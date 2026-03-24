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

class GaussianModel:
    """MIMOGS Gaussian scene model

    Learnable attributes per Gaussian:
    - mean                  : xyz
    - covariance            : rotation + scaling
    - opacity-like weight   :opacity
    """

    def __init__(
        self,
        target_gaussians: int = 50_000,
        optimizer_type: str = "default",
        device: str = "cuda",
        init_range: float = 5.0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.optimizer_type = optimizer_type
        self.target_gaussians = target_gaussians
        self.init_range = init_range

        self._xyz = torch.empty(0, device = self.device)
        self._scaling = torch.empty(0, device = self.device)
        self._rotation = torch.empty(0, device = self.device)
        self._opacity = torch.empty(0, device = self.device)
        self._gain_mag = torch.empty(0, device = self.device)

        self.optimizer = None
        self.xyz_scheduler_args = None

        self.xyz_gradient_accum = torch.empty(0,device = self.device)
        self.grad_denom = torch.empty(0,device = self.device)
        self.importance_accum = torch.empty(0,device = self.device)
        self.importance_denom = torch.empty(0,device = self.device)

        self.setup_functions()

    def setup_functions(self):
        self.scaling_activation = lambda x: torch.exp(torch.clamp(x, min=-10.0, max=5.0))
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = lambda x: F.normalize(x, dim=-1)

        self.gain_mag_activation = F.softplus
        self.gain_mag_inverse_activation = inverse_softplus

        self.covariance_activation = build_covariance_from_scaling_rotation


    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_gain_mag(self):
        return self.gain_mag_activation(self._gain_mag)

    @property
    def get_gain_weight(self):
        return self.get_opacity * self.get_gain_mag

    def get_covariance(self, scaling_modifier: float = 1.0):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self.get_rotation, return_strip = False
        )

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

        gain_mag_raw = self.gain_mag_inverse_activation(
            0.1 * torch.ones((n_points, 1), dtype = torch.float32, device = self.device)
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales_raw.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities_raw.requires_grad_(True))
        self._gain_mag = nn.Parameter(gain_mag_raw.requires_grad_(True))

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
            self._gain_mag.detach(),
            self.xyz_gradient_accum.detach(),
            self.grad_denom.detach(),
            self.importance_accum.detach(),
            self.importance_denom.detach(),
            None if self.optimizer is None else self.optimizer.state_dict(),
        )

    def restore(self, model_args, training_args):
        (
            self.target_gaussians,
            self.optimizer_type,
            self.init_range,
            xyz,
            scaling,
            rotation,
            opacity,
            gain_mag,
            xyz_gradient_accum,
            grad_denom,
            importance_accum,
            importance_denom,
            opt_dict,
        ) = model_args

        self._xyz = nn.Parameter(xyz.to(self.device).requires_grad_(True))
        self._scaling = nn.Parameter(scaling.to(self.device).requires_grad_(True))
        self._rotation = nn.Parameter(rotation.to(self.device).requires_grad_(True))
        self._opacity = nn.Parameter(opacity.to(self.device).requires_grad_(True))
        self._gain_mag = nn.Parameter(gain_mag.to(self.device).requires_grad_(True))

        self.training_setup(training_args)

        self.xyz_gradient_accum = xyz_gradient_accum.to(self.device)
        self.grad_denom = grad_denom.to(self.device)
        self.importance_accum = importance_accum.to(self.device)
        self.importance_denom = importance_denom.to(self.device)
        
        if opt_dict is not None:
            self.optimizer.load_state_dict(opt_dict)

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
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
            {"params": [self._gain_mag], "lr": training_args.gain_lr, "name": "gain_mag"},
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

        self.gain_scheduler_args = get_expon_lr_func(
            lr_init=training_args.gain_lr,
            lr_final=training_args.gain_lr_final,
            lr_delay_mult=1.0,
            max_steps=training_args.iterations,
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "opacity":
                lr = self.opacity_scheduler_args(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "gain_mag":
                lr = self.gain_scheduler_args(iteration)
                param_group["lr"] = lr

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
        attrs += ["gain_mag"]
        return attrs

    def save_ply(self, path: str):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        opacities = self.get_opacity.detach().cpu().numpy()
        scales = self.get_scaling.detach().cpu().numpy()
        rotations = self.get_rotation.detach().cpu().numpy()
        gain_mag = self.get_gain_mag.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            [xyz, normals, opacities, scales, rotations, gain_mag], axis = 1
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

        gain_mag = np.asarray(plydata.elements[0]["gain_mag"])[..., np.newaxis]

        xyz_t = torch.tensor(xyz, dtype=torch.float32, device=self.device)
        opacity_t = torch.tensor(opacities, dtype=torch.float32, device=self.device)
        scale_t = torch.tensor(scales, dtype=torch.float32, device=self.device)
        rot_t = torch.tensor(rots, dtype=torch.float32, device=self.device)
        gain_mag_t = torch.tensor(gain_mag, dtype=torch.float32, device=self.device)

        self._xyz = nn.Parameter(xyz_t.requires_grad_(True))
        self._opacity = nn.Parameter(
            self.inverse_opacity_activation(opacity_t).requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            self.scaling_inverse_activation(torch.clamp(scale_t, min=1e-8)).requires_grad_(True)
        )
        self._rotation = nn.Parameter(rot_t.requires_grad_(True))
        self._gain_mag = nn.Parameter(
            self.gain_mag_inverse_activation(torch.clamp(gain_mag_t, min=1e-8)).requires_grad_(True)
        )

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
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._gain_mag = optimizable_tensors["gain_mag"]

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
        new_gain_mag: torch.Tensor,
    ):
        tensors_dict = {
            "xyz": new_xyz,
            "opacity": new_opacity,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "gain_mag": new_gain_mag,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(tensors_dict)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._gain_mag = optimizable_tensors["gain_mag"]

        self._reset_statistics()

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
            self.get_scaling.max(dim=1).values <= clone_scale_threshold,
        )

        if importance_threshold > 0.0:
            selected_pts_mask = torch.logical_and(
                selected_pts_mask, avg_importance >= importance_threshold
            )

        if selected_pts_mask.sum() == 0:
            return

        new_xyz = self._xyz[selected_pts_mask].clone()
        new_opacity = self._opacity[selected_pts_mask].clone()
        new_scaling = self._scaling[selected_pts_mask].clone()
        new_rotation = self._rotation[selected_pts_mask].clone()
        new_gain_mag = self._gain_mag[selected_pts_mask].clone()


        self.densification_postfix(
            new_xyz, new_opacity, new_scaling, new_rotation, new_gain_mag
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
            self.get_scaling.max(dim=1).values > split_scale_threshold,
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
        new_xyz = torch.bmm(rot_mats, samples.unsqueeze(-1)).squeeze(-1) + base_xyz

        new_scaling = self.scaling_inverse_activation(
            torch.clamp(
                self.get_scaling[selected_pts_mask].repeat(n_splits, 1) / (0.8 * n_splits),
                min=1e-8,
            )
        )
        new_rotation = self.get_rotation[selected_pts_mask].repeat(n_splits, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(n_splits, 1)
        new_gain_mag = self._gain_mag[selected_pts_mask].repeat(n_splits, 1)

        self.densification_postfix(
            new_xyz, new_opacity, new_scaling, new_rotation, new_gain_mag
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
        min_gain_mag: float,
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
        prune_mask = torch.logical_or(
            prune_mask, self.get_gain_mag.squeeze(-1) < min_gain_mag
        )

        if max_scale is not None:
            prune_mask = torch.logical_or(
                prune_mask, self.get_scaling.max(dim=1).values > max_scale
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