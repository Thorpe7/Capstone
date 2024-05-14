""" Run script for training ResNet model, & saving training/testing results"""

import logging as log
from pathlib import Path
import torch
import pandas as pd
import torchvision
from torchvision import transforms
from torch.utils.data import random_split
from src.dataset_formatting import curate_torch_csv
from src.dataloader import custom_dataloader
from src.resnet_arch import ResNet, BottleNeckBlock, ResNetBlock
from src.plain_cnn import PlainCNN, CNNBlock
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
cifar_train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=(32, 32), antialias=True),
        transforms.RandomCrop(126, pad_if_needed=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

cifar_test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=(32, 32), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Create CIFAR dataset
train_size = 45000
valid_size = 5000
cifar_full_trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=None
)
cifar_trainset, cifar_validset = random_split(
    cifar_full_trainset, [train_size, valid_size]
)
cifar_trainset.dataset.transform = cifar_train_transform
cifar_validset.dataset.transform = cifar_test_transform


# cifar train & valid loaders
train_loader = torch.utils.data.DataLoader(
    cifar_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
valid_loader = torch.utils.data.DataLoader(
    cifar_validset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

# cifar test loader
cifar_testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=cifar_test_transform
)
test_loader = torch.utils.data.DataLoader(
    cifar_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
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
}
model = model_dict["resnet50"]

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

torch.save(model.state_dict(), f"{Path.home()}/git_repos/Capstone/Plain34_90e_cifar.pt")

# Plot ROC curve
y_test, y_score = create_test_score_list(trained_model, test_loader)
fpr, tpr, roc_auc = one_vs_rest_roc_calc(y_test, y_score, classes)
roc_plt = plot_roc_auc(fpr, tpr, roc_auc, classes)
roc_plt.savefig("results/figures/one_vs_rest_figure.png")
