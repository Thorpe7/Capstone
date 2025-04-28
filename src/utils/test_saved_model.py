import os
import sys
import logging as log
import torch
import torchvision
from torchvision import transforms

cwd = os.getcwd()
sys.path.append(os.path.join(cwd, "src"))

from resnet_arch import ResNet, BottleNeckBlock, ResNetBlock
from plain_cnn import PlainCNN, CNNBlock
from model_train import test_model
from dataloader import custom_dataloader

log.getLogger(__name__)
log.basicConfig(level=log.INFO)

# Check gpu
log.info("Checking machine gpu resources...")
log.info(f"{torch.cuda.is_available()}...")
log.info(f"{torch.cuda.get_device_name()}...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters
NUM_EPOCHS = 182
BATCH_SIZE = 128
LEARNING_RATE = 0.1

classes_dict = {
    "brain_classes": ("glial", "mengi", "none", "pituitary"),
    "cifar_classes": (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ),
}

# Select what classes used for testing
cifar_flag = True
if cifar_flag:
    test_classes = classes_dict["cifar_classes"]
    img_channels = 3

    cifar_test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(32, 32), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    cifar_testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=cifar_test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        cifar_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

else:
    test_classes = classes_dict["brain_classes"]
    img_channels = 1

    # Image transform and dataloader
    testing_data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(128, 128), antialias=True),
            transforms.Grayscale(1),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    test_dataset, test_loader = custom_dataloader(
        "data/v1/Testing/",
        transform=testing_data_transform,
        testing_flag=True,
        batch_size=BATCH_SIZE,
    )

# Check dataloading
test_iter = iter(test_loader)
image, label = next(test_iter)
log.info(f"Test Data dimension found: {image.shape}...")
log.info(f"Test Label dimension found: {label.shape}...")


# Select model architecture
model_dict = {
    "resnet18": ResNet(
        ResNetBlock,
        [2, 2, 2, 2],
        image_channels=img_channels,
        num_classes=len(test_classes),
        bottleneck=False,
    ).to(DEVICE),
    "resnet34": ResNet(
        ResNetBlock,
        [3, 4, 6, 3],
        image_channels=img_channels,
        num_classes=len(test_classes),
        bottleneck=False,
    ).to(DEVICE),
    "resnet50": ResNet(
        BottleNeckBlock,
        [3, 4, 6, 3],
        image_channels=img_channels,
        num_classes=len(test_classes),
        bottleneck=True,
    ).to(DEVICE),
    "plaincnn18": PlainCNN(
        CNNBlock,
        [3, 4, 6, 3],
        image_channels=img_channels,
        num_classes=len(test_classes),
    ).to(DEVICE),
    "plaincnn34": PlainCNN(
        CNNBlock,
        [6, 8, 12, 6],
        image_channels=img_channels,
        num_classes=len(test_classes),
    ).to(DEVICE),
}

model_arch = model_dict["plaincnn34"]
model_arch.load_state_dict(
    torch.load("/home/thorpe/git_repos/Capstone/Plain34_90e_cifar.pt")
)

# Test the model
test_model(test_loader, DEVICE, model_arch, BATCH_SIZE, test_classes)
