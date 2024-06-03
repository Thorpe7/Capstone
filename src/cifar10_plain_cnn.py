import torch
import torch.nn as nn
import logging


class CifarCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(CifarCNNBlock, self).__init__()

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


class CifarPlainCNN(nn.Module):
    def __init__(
        self, Block: nn.Module, num_layers: list, image_channels: int, num_classes: int
    ):
        super(CifarPlainCNN, self).__init__()

        # First conv layer
        self.in_channels = 16
        self.conv1 = nn.Conv2d(
            image_channels, out_channels=16, kernel_size=3, stride=1, padding=3
        )
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # Call CNN Blocks
        self.block2 = self._make_layers(Block, num_layers[0], out_channels=16, stride=1)
        self.block3 = self._make_layers(Block, num_layers[1], out_channels=32, stride=2)
        self.block4 = self._make_layers(Block, num_layers[2], out_channels=64, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # layers contained in blocks 2-4 (6(n)) number of layers
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Dense Layer
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layers(self, Block, num_residual_blocks, out_channels, stride):
        layers = []
        layers.append(Block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        for _ in range(1, num_residual_blocks):
            block_value = Block(self.in_channels, out_channels, 1)
            layers.append(block_value)

        # Returns unpacked list
        return nn.Sequential(*layers)


if __name__ == "__main__":
    test_CNN = CifarPlainCNN(CifarCNNBlock, [6, 6, 6], 3, 10)
    print(test_CNN)
