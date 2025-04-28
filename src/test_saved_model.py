import os
import sys
import logging as log
import torch
import torchvision

from PIL import Image
from torchvision import transforms
from dataset_formatting import (
    calc_imgchnnl_mean_std,
    calculate_per_pixel_mean,
    SubtractPixelMean,
)
from cifar10_plain_cnn import CifarCNNBlock, CifarPlainCNN
from cifar10_resnet_arch import CifarResNet, CifarResNetBlock
from cifar10_plain_cnn_pooling import CifarPlainCNNwPool, CifarCNNBlockwPool
from cifar10_resnet_arch_pooling import CifarResNetwPool, CifarResNetBlockwPool
from dataloader import create_datasets

cwd = os.getcwd()
sys.path.append(os.path.join(cwd, "src"))

log.getLogger(__name__)
log.basicConfig(level=log.INFO)

# Check gpu
log.info("Checking machine gpu resources...")
log.info(f"{torch.cuda.is_available()}...")
log.info(f"{torch.cuda.get_device_name()}...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")

# Hyper parameters
NUM_EPOCHS = 182
BATCH_SIZE = 128
LEARNING_RATE = 0.1

# CT DATASET AND PYTORCH DATA LOADING
brain_classes = ("glial", "mengi", "none", "pituitary")
cifar_classes = (
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
)


# Select model architecture
model_dict = {
    "cifar_plaincnn20": CifarPlainCNN(
        CifarCNNBlock, [6, 6, 6], image_channels=3, num_classes=10
    ).to(DEVICE),
    "cifar_plaincnn32": CifarPlainCNN(
        CifarCNNBlock, [10, 10, 10], image_channels=3, num_classes=10
    ).to(DEVICE),
    "cifar_plaincnn44": CifarPlainCNN(
        CifarCNNBlock, [14, 14, 14], image_channels=3, num_classes=10
    ).to(DEVICE),
    "cifar_plaincnn56": CifarPlainCNN(
        CifarCNNBlock, [18, 18, 18], image_channels=3, num_classes=10
    ).to(DEVICE),
    "cifar_resnet20": CifarResNet(
        CifarResNetBlock, [6, 6, 6], image_channels=3, num_classes=10
    ).to(DEVICE),
    "cifar_resnet32": CifarResNet(
        CifarResNetBlock, [10, 10, 10], image_channels=3, num_classes=10
    ).to(DEVICE),
    "cifar_resnet44": CifarResNet(
        CifarResNetBlock, [14, 14, 14], image_channels=3, num_classes=10
    ).to(DEVICE),
    "cifar_resnet56": CifarResNet(
        CifarResNetBlock, [18, 18, 18], image_channels=3, num_classes=10
    ).to(DEVICE),
    "plaincnn20_pooling": CifarPlainCNNwPool(
        CifarCNNBlockwPool, [6, 6, 6], image_channels=3, num_classes=10
    ).to(DEVICE),
    "plaincnn32_pooling": CifarPlainCNNwPool(
        CifarCNNBlockwPool, [10, 10, 10], image_channels=3, num_classes=10
    ).to(DEVICE),
    "plaincnn44_pooling": CifarPlainCNNwPool(
        CifarCNNBlockwPool, [14, 14, 14], image_channels=3, num_classes=10
    ).to(DEVICE),
    "plaincnn56_pooling": CifarPlainCNNwPool(
        CifarCNNBlockwPool, [18, 18, 18], image_channels=3, num_classes=10
    ).to(DEVICE),
    "resnet20_pooling": CifarResNetwPool(
        CifarResNetBlockwPool, [6, 6, 6], image_channels=3, num_classes=10
    ).to(DEVICE),
    "resnet32_pooling": CifarResNetwPool(
        CifarResNetBlockwPool, [10, 10, 10], image_channels=3, num_classes=10
    ).to(DEVICE),
    "resnet44_pooling": CifarResNetwPool(
        CifarResNetBlockwPool, [14, 14, 14], image_channels=3, num_classes=10
    ).to(DEVICE),
    "resnet56_pooling": CifarResNetwPool(
        CifarResNetBlockwPool, [18, 18, 18], image_channels=3, num_classes=10
    ).to(DEVICE),
}

model = model_dict["cifar_plaincnn20"]
model.load_state_dict(
    torch.load(
        "/home/thorpe/git_repos/Capstone/results/performance_metrics/brain/plain20/brain_cifar_plaincnn20_182e.pt"
    )
)
model = model.to(DEVICE)
model.eval()

glial = Image.open(
    "/home/thorpe/git_repos/Capstone/data/v1/Testing/glioma/Te-gl_0010.jpg"
)
menin = Image.open(
    "/home/thorpe/git_repos/Capstone/data/v1/Testing/meningioma/Te-me_0012.jpg"
)
healthy = Image.open(
    "/home/thorpe/git_repos/Capstone/data/v1/Testing/notumor/Te-no_0010.jpg"
)
pitu = Image.open(
    "/home/thorpe/git_repos/Capstone/data/v1/Testing/pituitary/Te-pi_0010.jpg"
)

test_dataset = create_datasets(
    "data/v1/Testing/",
    transform=None,
    testing_flag=True,
)
temp_transform = torchvision.transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((32, 32))]
)
testing_mean = calculate_per_pixel_mean(
    test_dataset, BATCH_SIZE, testing_flag=True, transform=temp_transform
)

testing_mean_tensor, testing_std_tensor = calc_imgchnnl_mean_std(
    test_dataset, BATCH_SIZE, test_flag=True, transform=temp_transform
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=(32, 32), antialias=True),
        SubtractPixelMean(testing_mean),
        transforms.Normalize(testing_mean_tensor, testing_std_tensor),
    ]
)

img_tensor = test_transform(menin)
img_tensor = img_tensor.unsqueeze(0)
img_tensor = img_tensor.to(DEVICE)

with torch.no_grad():
    output = model(img_tensor)

predicted_class = output.argmax(dim=1)
print(predicted_class.item())
