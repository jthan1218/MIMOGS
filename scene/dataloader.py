import os
import random

import imageio
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import Dataset

def split_dataset(datadir, ratio=0.1, dataset_type="rfid"):
    """Random shuffle train/test set"""
    if dataset_type == "rfid":
        spectrum_dir = os.path.join(datadir, "spectrum")
        spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith(".png")])
        index = [x.split(".")[0] for x in spt_names]
    
    train_len = int(len(index)*ratio)
    train_index = np.array(index[:train_len])
    test_index = np.array(index[train_len:])
    
    np.savetxt(os.path.join(datadir, "train_index.txt"), train_index, fmt="%s")
    np.savetxt(os.path.join(datadir, "test_index.txt"), test_index, fmt="%s")


class Spectrum_dataset(Dataset):
    """Spectrum dataset class."""

    def __init__(self, datadir, indexdir) -> None:
        super().__init__()
        self.datadir = datadir
        self.tx_pos_dir = os.path.join(datadir, "tx_pos.csv")
        self.spectrum_dir = os.path.join(datadir, "spectrum")
        self.spt_names = sorted([f for f in os.listdir(self.spectrum_dir) if f.endswith(".png")])
        self.dataset_index = np.loadtxt(indexdir, dtype=str)
        self.tx_pos = pd.read_csv(self.tx_pos_dir).values
        self.n_samples = len(self.dataset_index)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        img_name = os.path.join(self.spectrum_dir, self.dataset_index[index] + ".png")
        spectrum = imageio.imread(img_name) / 255.0
        spectrum = torch.tensor(spectrum, dtype=torch.float32)

        tx_pos_i = torch.tensor(
            self.tx_pos[int(self.dataset_index[index]) - 1], dtype=torch.float32
        )

        return spectrum, tx_pos_i

class DeepMIMODataset(Dataset):
    """DeepMIMO dataset from mat files with auto normalized positions."""

    def __init__(self, mat_path: str, normalize: bool = True) -> None:
        super().__init__()
        mat_data = sio.loadmat(mat_path)

        self.positions = torch.tensor(mat_data["positions"], dtype=torch.float32)
        self.magnitude = torch.tensor(mat_data["magnitude"], dtype=torch.float32)

        if self.positions.shape[0] != self.magnitude.shape[0]:
            raise ValueError("Positions and magnitude must have the same number of samples")

        self.n_samples = self.positions.shape[0]
        self.normalize = normalize
        self.scale_factor = 1.0

        if self.normalize:
            max_val = self.positions.abs().max()
            self.scale_factor = float(max_val) + 1e-6

            print(f"[Dataset] Auto-normalizing positions...")
            print(f"   - Max coordinate found: {max_val:.4f}")
            print(f"   - Scale factor applied: {self.scale_factor:.4f}")

            self.positions = self.positions / self.scale_factor

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.magnitude[index], self.positions[index]

class UmiDataset(Dataset):
    """UMi dataset from mat files with auto-normalized positions."""

    def __init__(self, mat_path: str, normalize: bool = True) -> None:
        super().__init__()
        mat_data = sio.loadmat(mat_path)

        self.positions = torch.tensor(mat_data["positions"], dtype=torch.float32)
        self.magnitude = torch.tensor(mat_data["magnitude"], dtype=torch.float32)

        if self.positions.shape[0] != self.magnitude.shape[0]:
            raise ValueError("Positions and magnitude must have the same number of samples")

        self.n_samples = self.positions.shape[0]
        self.normalize = normalize
        self.scale_factor = 1.0

        if self.normalize:
            max_val = self.positions.abs().max()
            self.scale_factor = float(max_val) + 1e-6

            print(f"[Dataset] Auto-normalizing positions...")
            print(f"   - Max coordinate found: {max_val:.4f}")
            print(f"   - Scale factor applied: {self.scale_factor:.4f}")

            self.positions = self.positions / self.scale_factor

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.magnitude[index], self.positions[index]

dataset_dict = {"rfid": Spectrum_dataset, "mimo": DeepMIMODataset, "mimo2": UmiDataset}
