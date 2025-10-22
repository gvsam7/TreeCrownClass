import torchvision
from torch import nn
from models.ResNet import ResNet18, ResNet50, ResNet101, ResNet152
from models.EfficientNet import EfficientNet
from models.ViT import ViT


def networks(architecture, in_channels, num_classes, pretrained, requires_grad, global_pooling, version, vit_cfg=None):
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
    elif architecture == 'vit':
        cfg = vit_cfg or {}
        model = ViT(
            img_size=cfg.get('img_size'),
            in_channels=in_channels,
            patch_size=cfg.get('patch_size'),
            hidden_size=cfg.get('hidden_size'),
            num_layers=cfg.get('num_layers'),
            num_heads=cfg.get('num_heads'),
            num_classes=num_classes
        )
    else:
        model = ResNet152(in_channels, num_classes)
    return model
