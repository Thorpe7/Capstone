""" Have the model predict on an image """
import torch
from torchvision import transforms
from resnet34_arch import ResNet, Block
from PIL import Image
from pathlib import Path

model = ResNet(Block, [2, 2, 2, 2], image_channels=1, num_classes=4)
model.load_state_dict(torch.load("/home/thorpe/Git/Capstone/model.pt"))
model.eval()

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


image = Image.open(
    Path("/home/thorpe/Git/Capstone/data_v1/Testing/notumor/Te-no_0123.jpg")
)
tran_image = data_transform(image)
tran_image = tran_image.view(1, 1, 128, 128)

output = model(tran_image)


prediction = int(torch.max(output.data, 1)[1].numpy())
print(prediction)
