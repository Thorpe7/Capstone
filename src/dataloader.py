"""
Creates the pytorch dataset & dataloader objects for use by model

"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets


def create_datasets(root_folder: str, transform, testing_flag: bool = False):
    """ """

    tmp_dataset = datasets.ImageFolder(root=root_folder, transform=transform)

    if testing_flag:
        return tmp_dataset
    else:
        tmp_train, tmp_valid = torch.utils.data.random_split(tmp_dataset, [3852, 1651])
        return tmp_train, tmp_valid


# [2009, 861] full training set
# [1733, 742] no non-tumors
# [856, 365] non tumor and glioma

# [4000,1712] v1 5712 total
# [3852,1651] cleaned v1 5503 total
# 5503
