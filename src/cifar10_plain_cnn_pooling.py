import torch
import torch.nn as nn
import logging
import torchvision
import torch.nn.functional as F

from torchview import draw_graph


class CifarCNNBlockwPool(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, pool_flag: bool = False):
        super(CifarCNNBlockwPool, self).__init__()
        self.pool_flag = pool_flag

        # Define block layers
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        # Apply relu
        x = self.relu(x)
        if self.pool_flag:
            x = self.max_pool(x)
        return x


class CifarPlainCNNwPool(nn.Module):
    def __init__(
        self, Block: nn.Module, num_layers: list, image_channels: int, num_classes: int
    ):
        super(CifarPlainCNNwPool, self).__init__()

        # First conv layer
        self.in_channels = 16
        self.conv1 = nn.Conv2d(
            image_channels, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        # Call CNN Blocks
        self.block1 = self._make_layers(Block, num_layers[0], out_channels=16, stride=1)
        self.block2 = self._make_layers(Block, num_layers[1], out_channels=32, stride=1)
        self.block3 = self._make_layers(Block, num_layers[2], out_channels=64, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)  #! Dropout layer
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # layers contained in blocks 2-4 (6(n)) number of layers
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Dense Layer
        x = self.avgpool(x)
        x = self.dropout(x)  #! Dropout layer
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)

        return x

    def _make_layers(self, Block, num_residual_blocks, out_channels, stride):
        layers = []
        layers.append(Block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        for val in range(1, num_residual_blocks):
            if val == max(range(1, num_residual_blocks)):
                block_value = Block(self.in_channels, out_channels, 1, pool_flag=True)

            else:
                block_value = Block(self.in_channels, out_channels, 1)

            layers.append(block_value)

        # Returns unpacked list
        return nn.Sequential(*layers)


if __name__ == "__main__":
    test_CNN = CifarPlainCNNwPool(CifarCNNBlockwPool, [6, 6, 6], 3, 10)

    model_graph = draw_graph(
        model=test_CNN,
        input_size=(1, 3, 32, 32),
        expand_nested=True,
        save_graph=True,
        filename="Cifar10_PlainCNN_Architecture",
    )
    model_graph.visual_graph

    vgg_model = torchvision.models.vgg19()

    vgg_graph = draw_graph(
        model=vgg_model,
        input_size=(1, 3, 32, 32),
        expand_nested=True,
        save_graph=True,
        filename="BASE_VGG_Architecture",
    )
    model_graph.visual_graph

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(test_CNN))
