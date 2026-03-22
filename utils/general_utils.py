import os
import sys
from datetime import datetime
import numpy as np
import random
import torch

def inverse_sigmoid(x, eps: float = 1e-6):
    x = torch.clamp(x, eps, 1.0- eps)
    return torch.log(x/(1.0-x))

def inverse_softplus(y, threshold: float = 20.0):
    """stable inverse of softplus function"""
    return torch.where(y > threshold, y, y+torch.log(-torch.expm1(-y)))

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels
    Continuous learning rate decay function. Adapted from JaxNeRF
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    device = L.device
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(torch.clamp(torch.sum(r*r, dim=1), min=1e-12))

    q = r / norm[:, None]

    device = r.device
    R = torch.zeros((q.size(0), 3, 3), device=device, dtype=r.dtype)

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - w*z)
    R[:, 0, 2] = 2 * (x*z + w*y)
    R[:, 1, 0] = 2 * (x*y + w*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - w*x)
    R[:, 2, 0] = 2 * (x*z - w*y)
    R[:, 2, 1] = 2 * (y*z + w*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):

    device = s.device
    L = torch.zeros((s.shape[0], 3, 3), dtype=s.dtype, device=device)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_covariance_from_scaling_rotation(
    scaling,
    scaling_modifier,
    rotation,
    return_strip: bool = False,
    ):
    """
    scaling: (N,3), expected to be positive
    rotation: (N,4), quaternion
    returns:
    - (N,3,3) covariance matrix if return_strip is False
    - (N,6) upper/lower symmetric packed form if return_strip is True
    """

    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1,2)
    if return_strip:
        return strip_symmetric(actual_covariance)
    return actual_covariance

def mkdir_p(path: str):
    if path:
        os.makedirs(path, exist_ok=True)

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # torch.cuda.set_device(torch.device("cuda:0"))