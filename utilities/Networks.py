from models.ResNet import ResNet18, ResNet50, ResNet101, ResNet152


def networks(architecture, in_channels, num_classes, pretrained, requires_grad, global_pooling):
    if architecture == 'resnet18':
        model = ResNet18(in_channels, num_classes)
    elif architecture == 'resnet50':
        model = ResNet50(in_channels, num_classes)
    elif architecture == 'resnet101':
        model = ResNet101(in_channels, num_classes)
    else:
        model = ResNet152(in_channels, num_classes)
    return model
