""" Run script for training ResNet model, & saving training/testing results"""
import logging as log
from pathlib import Path
import torch
from torchvision import transforms
from src.dataset_formatting import curate_torch_csv
from src.dataloader import custom_dataloader
from src.resnet50_arch import ResNet, Block
from src.model_train import (
    train_model,
    test_model,
    compute_accuracy,
    compute_confusion_matrix,
    plot_accuracy_per_iter,
)
from src.model_evaluation import (
    create_test_score_list,
    create_roc_calc_auc,
    plot_roc_auc,
)


log.getLogger(__name__)
log.basicConfig(level=log.INFO)

# Check gpu
log.info("Checking machine gpu resources...")
log.info(f"{torch.cuda.is_available()}...")
log.info(f"{torch.cuda.get_device_name()}...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters
NUM_EPOCHS = 2
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
data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=(128, 128), antialias=True),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Grayscale(1),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

# Create import data & create dataloader
train_dataset, valid_dataset, train_loader, valid_loader = custom_dataloader(
    "data/v1/Training/",
    transform=data_transform,
    testing_flag=False,
    batch_size=BATCH_SIZE,
)
test_dataset, test_loader = custom_dataloader(
    "data/v1/Testing/",
    transform=data_transform,
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

# Attempt different model sizes
# Resnet18, smaller model should generalize better
# model = ResNet(Block, [2, 2, 2, 2], image_channels=1, num_classes=4).to(DEVICE)

# Resnet50, original model size
model = ResNet(Block, [3, 4, 6, 3], image_channels=1, num_classes=4).to(DEVICE)

# Resnet101
# model = ResNet(Block, [3, 4, 23, 3], image_channels=1, num_classes=4).to(DEVICE)

trained_model, epoch_list, train_acc, valid_acc = train_model(
    model, train_loader, valid_loader, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, DEVICE
)

print(epoch_list, len(epoch_list))
print(train_acc, len(train_acc))
plot_accuracy_per_iter(epoch_list, train_acc, "Training Accuracy")
test_model(test_loader, DEVICE, trained_model, BATCH_SIZE, classes)

compute_accuracy(trained_model, train_loader, DEVICE, "Training")
mat = compute_confusion_matrix(trained_model, train_loader, DEVICE)
print(mat)

compute_accuracy(trained_model, valid_loader, DEVICE, "Validation")
mat = compute_confusion_matrix(trained_model, valid_loader, DEVICE)
print(mat)

compute_accuracy(trained_model, test_loader, DEVICE, "Testing")
mat = compute_confusion_matrix(trained_model, test_loader, DEVICE)
print(mat)

torch.save(model.state_dict(), f"{Path.home()}/git_repos/Capstone/model.pt")


y_test, y_score = create_test_score_list(model, test_loader)
fpr, tpr, thresholds, roc_auc = create_roc_calc_auc(y_test, y_score)
roc_plt = plot_roc_auc(fpr, tpr, thresholds, roc_auc)
roc_plt.show()
