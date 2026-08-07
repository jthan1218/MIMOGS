import math
import os
import yaml

from torch.utils.data import DataLoader

from utils.system_utils import searchForMaxIteration
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from scene.dataloader import *
import numpy as np
import torch

def square_array_shape(num_beams: int) -> tuple:
    """Factor a beam count into the most square (horizontal, vertical) UPA shape.

    The DFT beam grid is built from the array geometry, so the factorization has
    to match the array the dataset was generated with.  Every dataset in this
    repository uses a square array (4, 16, 64 and 100 beams), and a square shape
    is also the only one that cannot be silently transposed by swapping the two
    axes -- see ``_build_beam_uv_grid`` in the renderer.
    """
    num_beams = int(num_beams)
    if num_beams < 1:
        raise ValueError(f"beam count must be positive, got {num_beams}")

    horizontal = math.isqrt(num_beams)
    while horizontal > 1 and num_beams % horizontal:
        horizontal -= 1
    return (horizontal, num_beams // horizontal)


def build_power_balanced_weights(dataset, num_bins: int = 12):
    powers = (
        dataset.magnitude.float()
        .reshape(len(dataset), -1)
        .pow(2)
        .mean(dim=1)
        .cpu()
        .numpy()
    )

    logp = np.log10(np.maximum(powers, 1e-12))
    lo = float(logp.min())
    hi = float(logp.max())

    if hi - lo < 1e-12:
        return torch.ones(len(dataset), dtype=torch.double)

    edges = np.linspace(lo, hi, num_bins + 1)
    bin_ids = np.digitize(logp, edges[1:-1], right=False).astype(np.int64)
    counts = np.bincount(bin_ids, minlength=num_bins)

    weights = 1.0 / np.maximum(counts[bin_ids], 1)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.double)

class Scene:

    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration = None,
        shuffle = True,
        resolution_scales = [1.0],
        ):

        """
        MIMOGS Scene manager

        Responsibilities:
        - Keep dataset/dataloader handles
        - load BS metadata
        - optionally restore the latest saved Gaussian state
        - provide train/test iterators
        """

        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        self.batch_size = int(getattr(args, "batch_size", 8))
        self.num_epochs = int(getattr(args, "num_epochs", 10)) # no changes

        self.datadir = os.path.abspath(args.source_path)

        # BS metadata
        yaml_file_path = os.path.join(self.datadir, "bs_info.yml")
        with open(yaml_file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        self.bs_position = data["bs1"]["position"]
        self.bs_orientation = data["bs1"]["orientation"]

        self.r_o = self.bs_position
        self.gateway_orientation = self.bs_orientation

        dataset_name = data.get("dataset_name", "mimo")
        if dataset_name == "umi":
            dataset_key = "mimo2"
        else:
            dataset_key = "mimo"


        # Optional checkpoint loading
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))


        train_mat_path = os.path.join(self.datadir, "train.mat")
        test_mat_path = os.path.join(self.datadir, "test.mat")

        dataset_cls = dataset_dict[dataset_key]

        self.train_set = dataset_cls(train_mat_path)
        self.test_set = dataset_cls(test_mat_path)

        # The beam grid is dictated by the data: magnitude is (N, Nr, Nt).
        magnitude_shape = tuple(self.train_set.magnitude.shape)
        if len(magnitude_shape) != 3:
            raise ValueError(f"magnitude must be (N, Nr, Nt); got {magnitude_shape} in {train_mat_path}")

        self.beam_rows = int(magnitude_shape[1])
        self.beam_cols = int(magnitude_shape[2])

        # Beam grid.  "dft" derives the beam centers from the array shape, so
        # the shape must factor the beam count.  "custom_angles" reads the
        # centers from a measured steering codebook instead: the beam count is
        # len(az) * len(el), which need not be a UPA factorization at all, so
        # that path skips the shape resolution entirely.
        #
        # Imported here rather than at module scope: gaussian_renderer imports
        # scene.gaussian_model, so a top-level import would be circular.
        from gaussian_renderer import (
            MEASURED_BEAM_AZ_DEG,
            MEASURED_BEAM_EL_DEG,
            parse_angle_list,
        )

        self.beam_grid_mode = str(getattr(args, "beam_grid_mode", "dft") or "dft").lower()
        self.beam_az_deg = parse_angle_list(getattr(args, "beam_az_deg", ""), MEASURED_BEAM_AZ_DEG)
        self.beam_el_deg = parse_angle_list(getattr(args, "beam_el_deg", ""), MEASURED_BEAM_EL_DEG)

        if self.beam_grid_mode == "custom_angles":
            num_custom_beams = len(self.beam_az_deg) * len(self.beam_el_deg)
            for side, num_beams in (("rx", self.beam_rows), ("tx", self.beam_cols)):
                if num_custom_beams != num_beams:
                    raise ValueError(
                        f"beam_grid_mode=custom_angles gives {len(self.beam_az_deg)} x "
                        f"{len(self.beam_el_deg)} = {num_custom_beams} beams, but the dataset "
                        f"expects {num_beams} on the {side} side."
                    )
            # Descriptive only on this path: the renderer builds its centers
            # from the angle lists and never reads these.
            self.rx_shape = (len(self.beam_az_deg), len(self.beam_el_deg))
            self.tx_shape = self.rx_shape
        else:
            # Array shapes default to the square factorization of the beam
            # counts.  Non-square UPAs are declared through the override args.
            self.rx_shape = self._resolve_array_shape(args, "rx", self.beam_rows)
            self.tx_shape = self._resolve_array_shape(args, "tx", self.beam_cols)

        self.train_iter = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=int(getattr(args, "num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        self.test_iter = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=int(getattr(args, "num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
            )

    @staticmethod
    def _resolve_array_shape(args, side: str, num_beams: int) -> tuple:
        """Return the ``(horizontal, vertical)`` array shape for one side.

        ``0`` means "unspecified", so the shape is derived from the beam count.
        """
        horizontal = int(getattr(args, f"{side}_shape_h", 0))
        vertical = int(getattr(args, f"{side}_shape_v", 0))

        if horizontal <= 0 or vertical <= 0:
            return square_array_shape(num_beams)

        if horizontal * vertical != num_beams:
            raise ValueError(
                f"--{side}_shape_h x --{side}_shape_v = {horizontal}x{vertical} = "
                f"{horizontal * vertical} beams, but the dataset expects {num_beams}."
            )

        return (horizontal, vertical)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        os.makedirs(point_cloud_path, exist_ok=True)
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def dataset_init(self):
        self.train_iter_dataset = iter(self.train_iter)
        self.test_iter_dataset = iter(self.test_iter)
        