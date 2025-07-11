import torch.nn as nn
from models.Block import Block, BasicBlock


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)  # 2048

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * block.expansion  # 64*4=256 -> assumming block.expansion=4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))  # 256 -> 64, o/p=64*4

        return nn.Sequential(*layers)


def ResNet18(in_channels, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels, num_classes)


def ResNet50(in_channels, num_classes):
    return ResNet(Block, [3, 4, 6, 3], in_channels, num_classes)


def ResNet101(in_channels, num_classes):
    return ResNet(Block, [3, 4, 23, 3], in_channels, num_classes)


def ResNet152(in_channels, num_classes):
    return ResNet(Block, [3, 8, 36, 3], in_channels, num_classes)

