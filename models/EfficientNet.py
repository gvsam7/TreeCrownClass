import torch
import torch.nn as nn
from math import ceil

base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3]
]

phi_values = {
    # (phi, resolution, drop_rate)
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5)
}


class SiLU(nn.Module):  # export-friendly swish
    def forward(self, x):
        return x * torch.sigmoid(x)


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super().__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        reduced_dim = max(1, reduced_dim)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio,
                 reduction=4, survival_prob=0.8):
        super().__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = int(in_channels * expand_ratio)
        self.expand = hidden_dim != in_channels
        reduced_dim = max(1, int(in_channels / reduction))

        if self.expand:
            self.expand_conv = CNNBlock(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)

        self.conv = nn.Sequential(
            CNNBlock(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def stochastic_depth(self, x):
        if not self.training or self.survival_prob == 1.0:
            return x
        binary_tensor = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob).float()
        return (x / self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs
        out = self.conv(x)
        if self.use_residual:
            return self.stochastic_depth(out) + inputs
        else:
            return out


class EfficientNet(nn.Module):
    def __init__(self, version, in_channels, num_classes,
                 pretrained=False, requires_grad=True, global_pooling='avg'):
        super().__init__()
        self.in_channels = in_channels
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = int(ceil(1280 * width_factor))

        # pooling choice
        self.pool = nn.AdaptiveAvgPool2d(1) if global_pooling == 'avg' else nn.Identity()

        # pass in_channels into feature builder
        self.features = self.create_features(width_factor, depth_factor, last_channels, in_channels=self.in_channels)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes)
        )

        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False

        # pretrained loading left to caller (see earlier helper if needed)

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        """
        Returns (width_factor, depth_factor, dropout_rate) for a given version key.
        Must be indented inside the EfficientNet class.
        """
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels, in_channels=3):
        init_channels = int(32 * width_factor)
        # use the provided in_channels here
        features = [CNNBlock(in_channels, init_channels, 3, stride=2, padding=1)]
        in_c = init_channels

        for expand_ratio, c, repeats, stride, kernel_size in base_model:
            out_channels = int(4 * ceil(int(c * width_factor) / 4))
            layers_repeats = int(ceil(repeats * depth_factor))

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels=in_c,
                        out_channels=out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    )
                )
                in_c = out_channels

        features.append(CNNBlock(in_c, last_channels, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*features)

