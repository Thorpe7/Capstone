""" Run script for training ResNet model, & saving training/testing results"""

import logging as log
from pathlib import Path
import torch
import pandas as pd
import torchvision
from torchvision import transforms
from torch.utils.data import random_split
from src.dataset_formatting import (
    calc_imgchnnl_mean_std,
    calculate_per_pixel_mean,
    SubtractPixelMean,
)
from src.resnet_arch import ResNet, BottleNeckBlock, ResNetBlock
from src.plain_cnn import PlainCNN, CNNBlock
from src.cifar10_plain_cnn import CifarCNNBlock, CifarPlainCNN
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

# run defualt resnet from torchvision to validate personal model
# from torchvision.models import resnet18, ResNet18_Weights

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

# CIFAR Transforms
classes = (
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

# Create CIFAR dataset train and validation sets
train_size = 45000
valid_size = 5000
cifar_full_trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=None
)
cifar_trainset, cifar_validset = random_split(
    cifar_full_trainset, [train_size, valid_size]
)
# cifar test loader
cifar_testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=None
)

# Calculate per pixel mean of CIFAR training, validation, and testing datasets
training_mean = calculate_per_pixel_mean(cifar_trainset, BATCH_SIZE)
valid_mean = calculate_per_pixel_mean(cifar_validset, BATCH_SIZE, testing_flag=True)
testing_mean = calculate_per_pixel_mean(cifar_testset, BATCH_SIZE, testing_flag=True)

# Calculate mean and std of CIFAR dataset
training_mean_tensor, training_std_tensor = calc_imgchnnl_mean_std(
    cifar_trainset, BATCH_SIZE
)  # mean: tensor([0.1638, 0.1607, 0.1488]), std: tensor([0.2719, 0.2671, 0.2589])
valid_mean_tensor, valid_std_tensor = calc_imgchnnl_mean_std(cifar_validset, BATCH_SIZE)
testing_mean_tensor, testing_std_tensor = calc_imgchnnl_mean_std(
    cifar_testset, BATCH_SIZE, test_flag=True
)

# Create CIFAR transforms & apply
cifar_train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        SubtractPixelMean(training_mean),
        transforms.Normalize(training_mean_tensor, training_std_tensor),
        transforms.Pad(padding=4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(32, pad_if_needed=True),
    ]
)

cifar_valid_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        SubtractPixelMean(valid_mean),
        transforms.Normalize(valid_mean_tensor, valid_std_tensor),
    ]
)
cifar_test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        SubtractPixelMean(testing_mean),
        transforms.Normalize(testing_mean_tensor, testing_std_tensor),
    ]
)

cifar_trainset.dataset.transform = cifar_train_transform
cifar_validset.dataset.transform = cifar_valid_transform
cifar_test_transform.transform = cifar_test_transform

# cifar train & valid loaders
train_loader = torch.utils.data.DataLoader(
    cifar_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
valid_loader = torch.utils.data.DataLoader(
    cifar_validset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)
test_loader = torch.utils.data.DataLoader(
    cifar_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

# Print quick dimension check
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
}
model = model_dict["cifar_plaincnn32"]
# model = resnet18(weights=None).to(DEVICE)  # check against default model

(
    trained_model,
    epoch_list,
    train_acc,
    valid_acc,
    train_loss_list,
    valid_loss_list,
) = train_model(model, train_loader, valid_loader, NUM_EPOCHS, LEARNING_RATE, DEVICE)

####################

# Save results, create figures and tables
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
    f"{Path.home()}/git_repos/Capstone/cifar_plaincnn44_182e_cifar.pt",
)

# Plot ROC curve
y_test, y_score = create_test_score_list(trained_model, test_loader)
fpr, tpr, roc_auc = one_vs_rest_roc_calc(y_test, y_score, classes)
roc_plt = plot_roc_auc(fpr, tpr, roc_auc, classes)
roc_plt.savefig("results/figures/one_vs_rest_figure.png")
