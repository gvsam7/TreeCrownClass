import torchvision
from torch import nn
from models.ResNet import ResNet18, ResNet50, ResNet101, ResNet152
from models.EfficientNet import EfficientNet
from models.ViT import ViT
from models.GaborLayer import GaborConv2d
from models.MixPool import MixPool
from models.Block import DACBlock


def replace_with_mixpool(module, alpha=0.6, learnable=False):
    """
    Recursively replace nn.MaxPool2d / nn.AvgPool2d with MixPool.
    Returns number of replacements.
    """
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.MaxPool2d, nn.AvgPool2d)):
            k = child.kernel_size
            s = child.stride
            p = child.padding
            # MixPool signature used here: MixPool(kernel_size, stride, padding, alpha)
            # if the MixPool supports a learnable flag use MixPool(..., alpha, learnable=learnable)
            try:
                mixed = MixPool(kernel_size=k, stride=s, padding=p, alpha=alpha)
            except TypeError:
                # fallback if MixPool signature differs
                mixed = MixPool(k, s, p, alpha)
            setattr(module, name, mixed)
            replaced += 1
        else:
            replaced += replace_with_mixpool(child, alpha=alpha, learnable=learnable)
    return replaced


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
    elif architecture == 'dilgabmpdensenet':
        model = torchvision.models.densenet161(pretrained=False)

        # get original conv out channels BEFORE replacing conv
        orig_out = model.features.conv0.out_channels

        model.features.conv0 = GaborConv2d(
            in_channels=in_channels,
            out_channels=orig_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # replace pooling layers with MixPool
        n_replaced = replace_with_mixpool(model, alpha=0.6, learnable=False)
        print(f"Replaced {n_replaced} pooling layers with MixPool")

        # append averaged dilated conv block (keeps channel count)
        model.features.add_module('avg_dilated', DACBlock(orig_out, out_planes=orig_out))

        # replace classifier head
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
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
