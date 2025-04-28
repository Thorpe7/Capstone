""" Run script for training ResNet model, & saving training/testing results"""

import logging as log
from pathlib import Path
import torch
import pandas as pd
from torchvision import transforms
from src.dataset_formatting import curate_torch_csv
from src.dataloader import custom_dataloader
from src.dropout_models.resnet50_arch_w_dropout import ResNetwDropout, Block
from src.dropout_models.plain_cnn_w_dropout import PlainCNNwDropout, CNNBlock
from src.model_train import (
    train_model,
    test_model,
    compute_accuracy,
    compute_confusion_matrix,
    plot_accuracy_per_iter,
    plot_confusion_matrix,
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
NUM_EPOCHS = 25
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
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


# Create transform for images
training_data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=(128, 128), antialias=True),
        transforms.RandomCrop(126, pad_if_needed=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Grayscale(1),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

testing_data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=(128, 128), antialias=True),
        transforms.Grayscale(1),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

# Create dataset
train_dataset, valid_dataset, train_loader, valid_loader = custom_dataloader(
    "data/v1/Training/",
    transform=training_data_transform,
    testing_flag=False,
    batch_size=BATCH_SIZE,
    validation_flag=True,
    validation_transform=testing_data_transform,
)
test_dataset, test_loader = custom_dataloader(
    "data/v1/Testing/",
    transform=testing_data_transform,
    testing_flag=True,
    batch_size=BATCH_SIZE,
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

### Resnet50, original model size
model = ResNetwDropout(Block, [3, 4, 6, 3], image_channels=1, num_classes=4).to(DEVICE)

### Tradition CNN
# model = PlainCNNwDropout(CNNBlock, [6, 8, 12, 6], image_channels=1, num_classes=4).to(DEVICE)

trained_model, epoch_list, train_acc, valid_acc, loss_list = train_model(
    model, train_loader, valid_loader, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, DEVICE
)

train_acc_plt = plot_accuracy_per_iter(epoch_list, train_acc, "Training Accuracy")
train_acc_plt.savefig("results/figures/train_acc.png")
valid_acc_lot = plot_accuracy_per_iter(epoch_list, valid_acc, "Validation Accuracy")
valid_acc_lot.savefig("results/figures/valid_acc.png")
loss_plt = plot_loss_curve(loss_list)
loss_plt.savefig("results/figures/loss_curve.png")

# Load model
# trained_model = ResNet(Block, [3, 4, 6, 3], image_channels=1, num_classes=4).to(DEVICE)
# # trained_model = PlainCNN(CNNBlock, [6, 8, 12, 6], image_channels=1, num_classes=4).to(DEVICE)
# trained_model.load_state_dict(
#     torch.load(
#         "/home/thorpe/git_repos/Capstone/results/saved_models/20_epochs/ResNet50_model_84.pt"
#     )
# )
trained_model.eval()

test_model(test_loader, DEVICE, trained_model, BATCH_SIZE, classes)

compute_accuracy(trained_model, train_loader, DEVICE, "Training")
mat = compute_confusion_matrix(trained_model, train_loader, DEVICE)
confusion_matrix_fig = plot_confusion_matrix(matrix=mat, classes=classes)
confusion_matrix_fig.savefig("results/figures/training_conf_mat.png")
metrics_df = get_performance_metrics(mat, classes)
metrics_df.to_csv("results/performance_metrics/training_conf_metrics.csv")

compute_accuracy(trained_model, valid_loader, DEVICE, "Validation")
mat = compute_confusion_matrix(trained_model, valid_loader, DEVICE)
confusion_matrix_fig = plot_confusion_matrix(matrix=mat, classes=classes)
confusion_matrix_fig.savefig("results/figures/valid_conf_mat.png")
metrics_df = get_performance_metrics(mat, classes)
metrics_df.to_csv("results/performance_metrics/validation_conf_metrics.csv")

compute_accuracy(trained_model, test_loader, DEVICE, "Testing")
mat = compute_confusion_matrix(trained_model, test_loader, DEVICE)
confusion_matrix_fig = plot_confusion_matrix(matrix=mat, classes=classes)
confusion_matrix_fig.savefig("results/figures/testing_conf_mat.png")
metrics_df = get_performance_metrics(mat, classes)
metrics_df.to_csv("results/performance_metrics/testing_conf_metrics.csv")


torch.save(model.state_dict(), f"{Path.home()}/git_repos/Capstone/PlainCNN_.pt")


y_test, y_score = create_test_score_list(trained_model, test_loader)
fpr, tpr, roc_auc = one_vs_rest_roc_calc(y_test, y_score, classes)
roc_plt = plot_roc_auc(fpr, tpr, roc_auc, classes)
roc_plt.savefig("results/figures/one_vs_rest_figure.png")
