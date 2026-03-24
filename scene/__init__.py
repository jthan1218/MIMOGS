import os
import yaml

from torch.utils.data import DataLoader

from utils.system_utils import searchForMaxIteration
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from scene.dataloader import *
import numpy as np
import torch

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

        self.batch_size = 1
        self.num_epochs = 20

        self.datadir = os.path.abspath(args.source_path)

        self.beam_rows = 4
        self.beam_cols = 16

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

        self.train_iter = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

        self.test_iter = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            )

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        os.makedirs(point_cloud_path, exist_ok=True)
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def dataset_init(self):
        self.train_iter_dataset = iter(self.train_iter)
        self.test_iter_dataset = iter(self.test_iter)
        