from __future__ import annotations

import torch
from torch import nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Light3DBackbone(nn.Module):
    def __init__(self, in_channels: int = 1, channels=(8, 16, 32)):
        super().__init__()
        layers = []
        prev = in_channels
        for idx, ch in enumerate(channels):
            layers.append(ConvBlock3D(prev, ch))
            if idx != len(channels) - 1:
                layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            prev = ch
        self.encoder = nn.Sequential(*layers)
        self.out_channels = prev

    def forward(self, x):
        return self.encoder(x)
