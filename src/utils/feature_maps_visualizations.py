import sys
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

cwd = os.getcwd()
sys.path.append(os.path.join(cwd, "src"))
from resnet_arch import ResNet, BottleNeckBlock, ResNetBlock
from plain_cnn import PlainCNN, CNNBlock

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model class (make sure it's defined or imported)
model = ResNet(ResNetBlock, [2, 2, 2, 2], image_channels=1, num_classes=4).to(DEVICE)

# Load the state dict
state_dict = torch.load("/home/thorpe/git_repos/Capstone/Plain_80epochs.pt")
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# for name,layer in model.named_modules():
#     print(name)

# Define a hook to capture the output of the feature maps
activation_maps = []


def get_activation_hook(module, input, output):
    activation_maps.append(output.detach())


# Register hook to the layer of interest (you need to know the layer name or access it directly)
layer_name = "layer3.0.conv2"  #'layer5.1'#'layer5.0.conv2'  # Example layer name, this will depend on your model's architecture
layer = dict([*model.named_modules()])[layer_name]
hook_handle = layer.register_forward_hook(get_activation_hook)

# Preprocess the image
image_path = "/home/thorpe/git_repos/Capstone/data/v1/Testing/meningioma/Te-me_0011.jpg"
image = Image.open(image_path)
testing_data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=(128, 128), antialias=True),
        transforms.Grayscale(1),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

input_tensor = testing_data_transform(image).unsqueeze(0).to(DEVICE)

# Forward pass through the model with the image
with torch.no_grad():
    model(input_tensor)

# Remove the hook
hook_handle.remove()

# Now visualize the feature maps
# Assuming we are interested in the first feature map
feature_map = activation_maps[0].squeeze(0)  # Remove batch dimension

# Plot the feature maps
# fig, axes = plt.subplots(int(feature_map.size(0) ** 0.5), int(feature_map.size(0) ** 0.5))
selected_feature_maps_indices = [0, 1]

# Modify the plotting code to only show the selected feature maps
fig, axes = plt.subplots(
    1, len(selected_feature_maps_indices)
)  # 1 row, N columns where N is the number of selected maps
if not isinstance(axes, np.ndarray):
    axes = [axes]  # If only one axis, put it in a list for consistency in iteration

for i, ax in enumerate(axes.flat):
    ax.imshow(
        feature_map[i].cpu().numpy(), cmap="viridis"
    )  # Feature maps can be large, so we plot a grid
    ax.axis("off")
plt.show()
