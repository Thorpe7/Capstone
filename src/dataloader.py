"""
Creates the pytorch dataset & dataloader objects for use by model

"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets


def custom_dataloader(
    root_folder: str,
    transform,
    testing_flag: bool = False,
    batch_size: int = 32,
    validation_flag: bool = False,
    validation_transform=None,
):
    """ """

    tmp_dataset = datasets.ImageFolder(root=root_folder, transform=transform)

    if testing_flag:
        tmp_data_loader = DataLoader(tmp_dataset, batch_size=batch_size, shuffle=False)
        return tmp_dataset, tmp_data_loader
    else:
        tmp_train, tmp_valid = torch.utils.data.random_split(tmp_dataset, [3852, 1651])
        if validation_flag:
            tmp_valid.transform = validation_transform
        tmp_data_loader = DataLoader(tmp_train, batch_size=batch_size, shuffle=True)
        tmp_valid_loader = DataLoader(tmp_valid, batch_size=batch_size, shuffle=True)
        return tmp_train, tmp_valid, tmp_data_loader, tmp_valid_loader


# [2009, 861] full training set
# [1733, 742] no non-tumors
# [856, 365] non tumor and glioma

# [4000,1712] v1 5712 total
# [3852,1651] cleaned v1 5503 total
