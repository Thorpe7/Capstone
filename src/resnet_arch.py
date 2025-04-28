import torch
import torch.nn as nn
import logging


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(ResNetBlock, self).__init__()
        self.expansion = 4

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
        x = self.conv2(x)
        x = self.bn2(x)

        # Implementation of residual
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        # Apply residual and relu
        x += identity
        x = self.relu(x)
        return x


class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(BottleNeckBlock, self).__init__()
        self.expansion = 4

        # Define block layers
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride, padding=0
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        # Implementation of residual
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        # Apply residual and relu
        x += identity
        x = self.relu(x)
        return x


class ResNet(
    nn.Module
):  # num_layers = how many time we will implement above block [list] e.g. [3,4,6,3]
    def __init__(
        self,
        Block: nn.Module,
        num_layers: list,
        image_channels: int,
        num_classes: int,
        bottleneck=False,
    ):
        super(ResNet, self).__init__()
        self.bottleneck = bottleneck

        # First conv layer
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Call Resnet Blocks
        self.layer2 = self._make_layer(
            Block, num_layers[0], out_channels=64, stride=1
        )  # 3 layers
        self.layer3 = self._make_layer(
            Block, num_layers[1], out_channels=128, stride=2
        )  # 4 layers
        self.layer4 = self._make_layer(
            Block, num_layers[2], out_channels=256, stride=2
        )  # 6 layers
        self.layer5 = self._make_layer(
            Block, num_layers[3], out_channels=512, stride=2
        )  # 3 layers

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if bottleneck:
            self.fc = nn.Linear(512 * 4, num_classes)
        else:
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

    def _make_layer(self, Block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if self.bottleneck:
            # When is identity layer needed
            if stride != 1 or self.in_channels != out_channels * 4:
                identity_downsample = nn.Sequential(
                    nn.Conv2d(
                        self.in_channels, out_channels * 4, kernel_size=1, stride=stride
                    ),
                    nn.BatchNorm2d(out_channels * 4),
                )
            # This is the layer that changes the number of channels
            layers.append(
                Block(self.in_channels, out_channels, identity_downsample, stride)
            )
            self.in_channels = out_channels * 4  #  64 * 4 = 256

            for i in range(num_residual_blocks - 1):
                layers.append(
                    Block(self.in_channels, out_channels)
                )  # input 256 -> 64, 64*4 (256) again

            # Returns unpacked list
            return nn.Sequential(*layers)

        else:
            # When is identity layer needed
            if stride != 1 or self.in_channels != out_channels:
                identity_downsample = nn.Sequential(
                    nn.Conv2d(
                        self.in_channels, out_channels, kernel_size=1, stride=stride
                    ),
                    nn.BatchNorm2d(out_channels),
                )
            # This is the layer that changes the number of channels
            layers.append(
                Block(self.in_channels, out_channels, identity_downsample, stride)
            )
            self.in_channels = out_channels  #  64 * 4 = 256

            for i in range(num_residual_blocks - 1):
                layers.append(
                    Block(self.in_channels, out_channels)
                )  # input 256 -> 64, 64*4 (256) again

            # Returns unpacked list
            return nn.Sequential(*layers)


# def ResNet50(img_channels=3, num_classes=4):
#     return ResNet(Block, [3,4,6,3], img_channels, num_classes)\

# def ResNet101(img_channels=3, num_classes=4):
#     return ResNet(Block, [3,4,23,3], img_channels, num_classes)

# def ResNet152(img_channels=3, num_classes=4):
#     return ResNet(Block, [3,4,36,3], img_channels, num_classes)
