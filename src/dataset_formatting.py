"""
Takes dataset and currates it for adequate use w/ pytorch dataset & loader

"""

import pathlib
from pathlib import Path
import pandas as pd
import logging as log
import shutil
from PIL import Image
import torch
import torchvision
from torch.utils.data import DataLoader
from typing import Tuple, Any
from torch import Tensor


log.basicConfig(level=log.INFO)
log = log.getLogger(__name__)


def curate_torch_csv(data_dir: pathlib.PosixPath, testing: bool = False) -> None:
    """
    Takes in a string path of data dir. Iterates through all sub directories.
    Creates w/ associated label. Also creates directory of all files in one directory.

    """
    log.info("Data curation started...")
    label = 0
    df_list = []
    if not data_dir.is_dir():
        log.error(f"{data_dir} is not valid directory...")

    # Iterate through data dir
    for item in data_dir.iterdir():
        tmp_df = pd.DataFrame()
        if item.is_dir():
            if not testing:
                file_paths = [x for x in item.glob("*.*") if x.is_file()]
            else:
                file_paths = [x for x in item.glob("*.*") if x.is_file()]
            tmp_df["images"] = file_paths
            tmp_df["label"] = label

        label += 1
        df_list.append(tmp_df)

    final_df = pd.concat(df_list, axis=0, ignore_index=True)
    final_df.to_csv(f"{data_dir}/{data_dir.name}.csv", index=False)


def find_smallest_img_dim(path_to_dir: pathlib.PosixPath) -> tuple:
    """

    Iterates through data directory and all subdirectories and finds smallest
    image dimesions to be used for image transformations.

    """
    min_dim = (3000, 3000)
    for item in path_to_dir.iterdir():
        if item.is_dir():
            min_dim = find_smallest_img_dim(item)
        else:
            if isinstance(item, pathlib.PosixPath):
                if item.is_file() and item.suffix == ".jpg":
                    img = Image.open(item)
                    img.size
                    if img.size < min_dim:
                        min_dim = img.size

    return min_dim


def calc_imgchnnl_mean_std(
    train_dataset: Any, batch_size: int, test_flag: bool = False
) -> Tuple[Tensor, Tensor]:
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_pixels = 0

    # Modify trianing dataset so images are tensors
    convert_to_tensor = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    if not test_flag:
        train_dataset.dataset.transform = convert_to_tensor
    else:
        train_dataset.transform = convert_to_tensor

    # Create dataloader
    tmp_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    for images, _ in tmp_loader:
        for i in range(3):
            mean[i] += images[:, i, :, :].sum()
            std[i] += images[:, i, :, :].pow(2).sum()
            total_pixels += images.size(0) * images.size(2) * images.size(3)

    # Calculate mean and std
    mean /= total_pixels
    std = torch.sqrt(std / total_pixels - mean**2)

    return mean, std


def calculate_per_pixel_mean(
    input_dataset: Any, batch_size: int, testing_flag: bool = False
) -> float:
    """Calculates the mean of all pixels in a given dataset.

    Args:
        input_dataset (Any): Pytorch dataloader object
        batch_size (int): Batch size for dataloader
        testing_flat (bool): Flag for testing dataset

    Returns:
        Tensor: Tensor of the mean pixel values
    """

    # Modify dataset so images are tensors
    convert_to_tensor = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    if not testing_flag:
        input_dataset.dataset.transform = convert_to_tensor
    else:
        input_dataset.transform = convert_to_tensor

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        input_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    mean = 0.0
    count = 0

    # Calculate mean of all batches
    for images, _ in dataloader:
        mean += images.sum(dim=0)
        count += images.size(0)

    mean /= count
    return mean


class SubtractPixelMean:
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, img):
        return img - self.mean
