import torch
import torch.nn as nn
import logging


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(CNNBlock, self).__init__()

        # Define block layers
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        # Apply relu
        x = self.relu(x)
        return x


class PlainCNN(
    nn.Module
):  # num_layers = how many layers for the block to create [6,8,12,6]
    def __init__(
        self, Block: nn.Module, num_layers: list, image_channels: int, num_classes: int
    ):
        super(PlainCNN, self).__init__()

        # First conv layer
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1
        )  # paper: (x,64,56,56), mine: (x,64,32,32)

        # Call Resnet Blocks
        self.layer2 = self._make_layers(
            Block, num_layers[0], out_channels=64, stride=1
        )  # 6 layers
        self.layer3 = self._make_layers(
            Block, num_layers[1], out_channels=128, stride=2
        )  # 8 layers
        self.layer4 = self._make_layers(
            Block, num_layers[2], out_channels=256, stride=2
        )  # 12 layers
        self.layer5 = self._make_layers(
            Block, num_layers[3], out_channels=512, stride=2
        )  # 6 layers

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # layers 2-5
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layers(self, Block, num_residual_blocks, out_channels, stride):
        layers = []

        for i in range(num_residual_blocks):
            block_value = Block(self.in_channels, out_channels, stride)
            if len(layers) == 0:
                self.in_channels = out_channels
                stride = 1
            layers.append(block_value)

        # Returns unpacked list
        return nn.Sequential(*layers)

    def calc_new_image_dims(
        self, in_channels: int, kernel_size: int, padding: int, stride: int
    ):
        value = ((in_channels + (2 * padding) - kernel_size) // stride) + 1
        return value


if __name__ == "__main__":
    TestCNN = PlainCNN(CNNBlock, [6, 8, 12, 6], 1, 4)
