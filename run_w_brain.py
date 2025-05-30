""" Run script for training ResNet model, & saving training/testing results"""

import logging as log
from pathlib import Path
import torch
import pandas as pd
import torchvision
from torchvision import transforms
from src.dataset_formatting import curate_torch_csv
from src.dataloader import create_datasets
from src.resnet_arch import ResNet, BottleNeckBlock, ResNetBlock
from src.cifar10_plain_cnn import CifarCNNBlock, CifarPlainCNN
from src.plain_cnn import PlainCNN, CNNBlock
from src.cifar10_resnet_arch import CifarResNet, CifarResNetBlock
from src.cifar10_plain_cnn_pooling import CifarPlainCNNwPool, CifarCNNBlockwPool
from src.cifar10_resnet_arch_pooling import CifarResNetwPool, CifarResNetBlockwPool
from src.dataset_formatting import (
    calc_imgchnnl_mean_std,
    calculate_per_pixel_mean,
    SubtractPixelMean,
)
from src.model_train import (
    train_model,
    test_model,
    compute_accuracy,
    compute_confusion_matrix,
    plot_accuracy_per_iter,
    plot_confusion_matrix,
    compute_error_rate,
)
from src.model_evaluation import (
    create_test_score_list,
    one_vs_rest_roc_calc,
    plot_roc_auc,
    plot_loss_curve,
    get_performance_metrics,
)

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


# CT DATASET AND PYTORCH DATA LOADING
classes = ("glial", "mengi", "none", "pituitary")

# Curate data
if Path("data/v1/Training/*.csv").exists():
    log.info(f"{Path('data/v1/Training/Training.csv')} already curated...")
else:
    curate_torch_csv(Path("data/v1/Training"))
    log.info(f"{Path('data/v1/Training/Training.csv')} created...")

if Path("data/v1/Testing/*.csv").exists():
    log.info(f"{Path('data/v1/Testing/Testings.csv')} already curated...")
else:
    curate_torch_csv(Path("data/v1/Testing"), testing=True)
    log.info(f"{Path('data/v1/Testing/Testing.csv')} created...")


# Create CT dataset
train_dataset, valid_dataset = create_datasets(
    "data/v1/Training/",
    transform=None,
    testing_flag=False,
)
test_dataset = create_datasets(
    "data/v1/Testing/",
    transform=None,
    testing_flag=True,
)

temp_transform = torchvision.transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((32, 32))]
)

# Calculate per pixel mean of training, validation, and testing datasets
training_mean = calculate_per_pixel_mean(
    train_dataset, BATCH_SIZE, transform=temp_transform
)
valid_mean = calculate_per_pixel_mean(
    valid_dataset, BATCH_SIZE, testing_flag=True, transform=temp_transform
)
testing_mean = calculate_per_pixel_mean(
    test_dataset, BATCH_SIZE, testing_flag=True, transform=temp_transform
)

# Calculate mean and std of dataset
training_mean_tensor, training_std_tensor = calc_imgchnnl_mean_std(
    train_dataset, BATCH_SIZE, transform=temp_transform
)  # mean: tensor([0.1638, 0.1607, 0.1488]), std: tensor([0.2719, 0.2671, 0.2589])
valid_mean_tensor, valid_std_tensor = calc_imgchnnl_mean_std(
    valid_dataset, BATCH_SIZE, transform=temp_transform
)
testing_mean_tensor, testing_std_tensor = calc_imgchnnl_mean_std(
    test_dataset, BATCH_SIZE, test_flag=True, transform=temp_transform
)

# Create transform for images
train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=(32, 32), antialias=True),
        SubtractPixelMean(training_mean),
        transforms.Normalize(training_mean_tensor, training_std_tensor),
        transforms.Pad(padding=4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(32, pad_if_needed=True),
    ]
)

valid_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=(32, 32), antialias=True),
        SubtractPixelMean(valid_mean),
        transforms.Normalize(valid_mean_tensor, valid_std_tensor),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=(32, 32), antialias=True),
        SubtractPixelMean(testing_mean),
        transforms.Normalize(testing_mean_tensor, testing_std_tensor),
    ]
)

# Apply transformations to dataset
train_dataset.dataset.transform = train_transform
valid_dataset.dataset.transform = valid_transform
test_dataset.transform = test_transform

# Create dataloaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)


iter_loader = iter(train_loader)
image, label = next(iter_loader)
log.info(f"Data dimension found: {image.shape}...")
log.info(f"Label dimension found: {label.shape}...")

valid_iter = iter(valid_loader)
image, label = next(valid_iter)
log.info(f"Valid Data dimension found: {image.shape}...")
log.info(f"Valid Label dimension found: {label.shape}...")

test_iter = iter(test_loader)
image, label = next(test_iter)
log.info(f"Test Data dimension found: {image.shape}...")
log.info(f"Test Label dimension found: {label.shape}...")


# Selecting model architecture
model_dict = {
    "resnet18": ResNet(
        ResNetBlock, [2, 2, 2, 2], image_channels=3, num_classes=10, bottleneck=False
    ).to(DEVICE),
    "resnet34": ResNet(
        ResNetBlock, [3, 4, 6, 3], image_channels=3, num_classes=10, bottleneck=False
    ).to(DEVICE),
    "resnet50": ResNet(
        BottleNeckBlock, [3, 4, 6, 3], image_channels=3, num_classes=10, bottleneck=True
    ).to(DEVICE),
    "plaincnn18": PlainCNN(CNNBlock, [3, 4, 6, 3], image_channels=3, num_classes=10).to(
        DEVICE
    ),
    "plaincnn34": PlainCNN(
        CNNBlock, [6, 8, 12, 6], image_channels=3, num_classes=10
    ).to(DEVICE),
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
    "resnet55_pooling": CifarResNetwPool(
        CifarResNetBlockwPool, [18, 18, 18], image_channels=3, num_classes=10
    ).to(DEVICE),
}
model_version = "plaincnn44_pooling"
model = model_dict[model_version]

(
    trained_model,
    epoch_list,
    train_acc,
    valid_acc,
    train_loss_list,
    valid_loss_list,
) = train_model(model, train_loader, valid_loader, NUM_EPOCHS, LEARNING_RATE, DEVICE)

train_acc_plt = plot_accuracy_per_iter(epoch_list, train_acc, "Training Accuracy")
train_acc_plt.savefig("results/figures/train_acc.png")
valid_acc_lot = plot_accuracy_per_iter(epoch_list, valid_acc, "Validation Accuracy")
valid_acc_lot.savefig("results/figures/valid_acc.png")
loss_plt = plot_loss_curve(train_loss_list, valid_loss_list)
loss_plt.savefig("results/figures/loss_curve.png")

loss_data = {
    "TrainingLoss": train_loss_list,
    "ValidationLoss": valid_loss_list,
    "Epochs": epoch_list,
}
loss_df = pd.DataFrame(loss_data)
loss_df.to_csv("results/performance_metrics/training_validation_loss.csv", index=False)

acc_data = {
    "TrainingAccuracy": train_acc,
    "ValidationAccuracy": valid_acc,
    "Epochs": epoch_list,
}
training_acc_df = pd.DataFrame(acc_data)
training_acc_df.to_csv(
    "results/performance_metrics/training_validation_acc.csv", index=False
)

# Test model
test_model(test_loader, DEVICE, trained_model, BATCH_SIZE, classes)

compute_accuracy(trained_model, train_loader, DEVICE, "Training", "post")
mat = compute_confusion_matrix(trained_model, train_loader, DEVICE)
confusion_matrix_fig = plot_confusion_matrix(matrix=mat, classes=classes)
confusion_matrix_fig.savefig("results/figures/training_conf_mat.png")
metrics_df = get_performance_metrics(mat, classes)
metrics_df.to_csv("results/performance_metrics/training_conf_metrics.csv")

compute_accuracy(trained_model, valid_loader, DEVICE, "Validation", "post")
mat = compute_confusion_matrix(trained_model, valid_loader, DEVICE)
confusion_matrix_fig = plot_confusion_matrix(matrix=mat, classes=classes)
confusion_matrix_fig.savefig("results/figures/valid_conf_mat.png")
metrics_df = get_performance_metrics(mat, classes)
metrics_df.to_csv("results/performance_metrics/validation_conf_metrics.csv")

compute_accuracy(trained_model, test_loader, DEVICE, "Testing", "post")
mat = compute_confusion_matrix(trained_model, test_loader, DEVICE)
confusion_matrix_fig = plot_confusion_matrix(matrix=mat, classes=classes)
confusion_matrix_fig.savefig("results/figures/testing_conf_mat.png")
metrics_df = get_performance_metrics(mat, classes)
metrics_df.to_csv("results/performance_metrics/testing_conf_metrics.csv")

compute_error_rate(trained_model, test_loader, DEVICE, "Testing")

torch.save(
    model.state_dict(),
    f"{Path.home()}/git_repos/Capstone/brain_dropout_pool_{model_version}_182e_01LR.pt",
)

# Plot ROC curve
y_test, y_score = create_test_score_list(trained_model, test_loader)
fpr, tpr, roc_auc = one_vs_rest_roc_calc(y_test, y_score, classes)
roc_plt = plot_roc_auc(fpr, tpr, roc_auc, classes)
roc_plt.savefig("results/figures/one_vs_rest_figure.png")
