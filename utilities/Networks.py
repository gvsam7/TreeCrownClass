import torchvision
from torch import nn
from models.ResNet import ResNet18, ResNet50, ResNet101, ResNet152
from models.EfficientNet import EfficientNet


def networks(architecture, in_channels, num_classes, pretrained, requires_grad, global_pooling, version):
    if architecture == 'resnet18':
        model = ResNet18(in_channels, num_classes)
    elif architecture == 'resnet50':
        model = ResNet50(in_channels, num_classes)
    elif architecture == 'resnet101':
        model = ResNet101(in_channels, num_classes)
    elif architecture == 'efficientnet':
        print(f"version: {version}")
        model = EfficientNet(version, in_channels, num_classes)
    elif architecture == 'densenet':
        model = torchvision.models.densenet161(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        for p in model.parameters():
            p.requires_grad = True  # default, ensures all layers train
    else:
        model = ResNet152(in_channels, num_classes)
    return model
