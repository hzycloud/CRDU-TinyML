import torch
from torch import nn


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.depthwise(inputs)
        outputs = self.pointwise(outputs)
        outputs = self.norm(outputs)
        return self.activation(outputs)


class DSCNN(nn.Module):
    def __init__(self, num_classes: int = 12, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            DepthwiseSeparableBlock(64, 64),
            nn.MaxPool2d(2),
            DepthwiseSeparableBlock(64, 128),
            nn.MaxPool2d(2),
            DepthwiseSeparableBlock(128, 128),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.features(inputs)
        outputs = self.pool(outputs).flatten(1)
        outputs = self.dropout(outputs)
        return self.classifier(outputs)


class AudioDSCNN(nn.Module):
    def __init__(self, num_classes: int = 12, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            DepthwiseSeparableBlock(64, 64),
            nn.MaxPool2d(2),
            DepthwiseSeparableBlock(64, 128),
            DepthwiseSeparableBlock(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.features(inputs)
        outputs = self.pool(outputs).flatten(1)
        outputs = self.dropout(outputs)
        return self.classifier(outputs)

