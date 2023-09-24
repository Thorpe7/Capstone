import torch
import logging as log
from pathlib import Path
from torchvision import transforms
from src.dataset_formatting import curate_torch_csv, find_smallest_img_dim
from src.dataloader import custom_dataloader
from src.resnet34_arch import ResNet, Block
from src.model_train import (
    train_model,
    test_model,
    compute_accuracy,
    compute_confusion_matrix,
)
from src.model_evaluation import (
    create_test_score_list,
    create_roc_calc_auc,
    plot_roc_auc,
)


log.getLogger(__name__)
log.basicConfig(level=log.INFO)

# Check gpu
log.info(f"Checking machine gpu resources...")
log.info(f"{torch.cuda.is_available()}...")
log.info(f"{torch.cuda.get_device_name()}...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters
num_epochs = 1
batch_size = 128
learning_rate = 0.0001
classes = ("glial", "mengi", "none", "pituitary")

# Curate data
if Path("data_v1/Training/*.csv").exists():
    log.info(f"{Path('data_v1/Training/Training.csv')} already curated...")
else:
    curate_torch_csv(Path("data_v1/Training"))
    log.info(f"{Path('data_v1/Training/Training.csv')} created...")

if Path("data_v1/Testing/*.csv").exists():
    log.info(f"{Path('data_v1/Testing/Testings.csv')} already curated...")
else:
    curate_torch_csv(Path("data_v1/Testing"), testing=True)
    log.info(f"{Path('data_v1/Testing/Testing.csv')} created...")

# Find smallest img dimesions in dataset
# dim_size = find_smallest_img_dim(path_to_dir=Path("data/"))
# log.info(f"Min image dims found: {dim_size}...")


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
    "data_v1/Training/",
    transform=data_transform,
    testing_flag=False,
    batch_size=batch_size,
)
test_dataset, test_loader = custom_dataloader(
    "data_v1/Testing/",
    transform=data_transform,
    testing_flag=True,
    batch_size=batch_size,
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
model = ResNet(Block, [2, 2, 2, 2], image_channels=1, num_classes=4).to(DEVICE)

# Resnet50, original model size
# model = ResNet(Block, [3, 4, 6, 3], image_channels=1, num_classes=4).to(DEVICE)

# Resnet101
# model = ResNet(Block, [3, 4, 23, 3], image_channels=1, num_classes=4).to(DEVICE)

trained_model = train_model(
    model, train_loader, valid_loader, num_epochs, batch_size, learning_rate, DEVICE
)

# test_model(test_loader, DEVICE, model,batch_size, classes)

compute_accuracy(trained_model, train_loader, DEVICE, "Training")
mat = compute_confusion_matrix(trained_model, train_loader, DEVICE)
print(mat)

compute_accuracy(trained_model, valid_loader, DEVICE, "Validation")
mat = compute_confusion_matrix(trained_model, valid_loader, DEVICE)
print(mat)

compute_accuracy(trained_model, test_loader, DEVICE, "Testing")
mat = compute_confusion_matrix(trained_model, test_loader, DEVICE)
print(mat)

torch.save(model.state_dict(), "/home/thorpe/Git/Capstone/model.pt")


y_test, y_score = create_test_score_list(model, test_loader)
fpr, tpr, thresholds, roc_auc = create_roc_calc_auc(y_test, y_score)
roc_plt = plot_roc_auc(fpr, tpr, thresholds, roc_auc)
roc_plt.show()
