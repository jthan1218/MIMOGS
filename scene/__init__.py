from ast import NodeTransformer
import os
import yaml

from torch.utils.data import DataLoader

from utils.system_utils import searchForMaxIteration
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from scene.dataloader import *

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

        self.model_path = args._model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        self.batch_size = 1
        self.num_epochs = 30

        self.datadir = os.path.abspath("./dataset/asu_campus_4by16_outdoor")

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

        # Optional checkpoint loading
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))


        train_mat_path = os.path.join(self.datadir, "train.mat")
        test_mat_path = os.path.join(self.datadir, "test.mat")

        self.train_set = DeepMIMODataset(train_mat_path)
        self.test_set = DeepMIMODataset(test_mat_path)

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
        