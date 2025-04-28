import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
from torchview import draw_graph


class CifarResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(CifarResNetBlock, self).__init__()

        # Define block layers
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # Implementation of residual
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        # Apply residual and relu
        x += identity
        x = self.relu(x)
        return x


class CifarResNet(nn.Module):
    def __init__(
        self,
        Block: nn.Module,
        num_layers: list,
        image_channels: int,
        num_classes: int,
    ):
        super(CifarResNet, self).__init__()

        # First conv layer, 3x3 kernel, 16 filters, output feature map of 32x32
        self.in_channels = 16
        self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        # Create Resnet Blocks
        self.block1 = self._make_layer(Block, num_layers[0], out_channels=16, stride=1)
        self.block2 = self._make_layer(Block, num_layers[1], out_channels=32, stride=2)
        self.block3 = self._make_layer(Block, num_layers[2], out_channels=64, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(p=0.5)  #! Dropout layer
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Resnet blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Dense Layer
        x = self.avgpool(x)
        # x = self.dropout(x)  #! Dropout layer
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)

        return x

    def _make_layer(self, Block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        # When is identity layer needed
        if stride != 1 or self.in_channels != out_channels:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

        # This is the layer that changes the number of channels
        layers.append(
            Block(self.in_channels, out_channels, identity_downsample, stride)
        )
        self.in_channels = out_channels

        for _ in range(num_residual_blocks - 1):
            layers.append(Block(self.in_channels, out_channels))

        # Returns unpacked list
        return nn.Sequential(*layers)


if __name__ == "__main__":
    TestNet = CifarResNet(
        CifarResNetBlock, [10, 10, 10], image_channels=3, num_classes=10
    )

    model_graph = draw_graph(
        model=TestNet,
        input_size=(1, 3, 32, 32),
        expand_nested=True,
        save_graph=True,
        filename="Cifar10_ResNet_Architecture",
    )
    model_graph.visual_graph

    resnet18 = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False)
    resnet18_graph = draw_graph(
        model=resnet18,
        input_size=(1, 3, 32, 32),
        expand_nested=True,
        save_graph=True,
        filename="BASE_ResNet18_Architecture",
    )

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"ResNet20 Model Parameters: {count_parameters(TestNet)}")
    # print(TestNet)
