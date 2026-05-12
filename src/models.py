"""Three CNN architectures of increasing complexity for FER2013 (48x48 grayscale, 7 classes)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 7
INPUT_SIZE = 48


ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def _act(name: str) -> nn.Module:
    cls = ACTIVATIONS.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(ACTIVATIONS)}")
    return cls(inplace=True) if cls in (nn.ReLU, nn.LeakyReLU, nn.ELU) else cls()


class SimpleCNN(nn.Module):
    """Model A — Shallow baseline. ~120k params."""

    def __init__(self, num_classes: int = NUM_CLASSES, activation: str = "relu", dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            _act(activation),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            _act(activation),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 12 * 12, 128),
            _act(activation),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class DeeperCNN(nn.Module):
    """Model B — VGG-style with BatchNorm + Dropout. ~1.5M params."""

    def __init__(self, num_classes: int = NUM_CLASSES, activation: str = "relu", dropout: float = 0.4):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                _act(activation),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                _act(activation),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, 64),
            block(64, 128),
            block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 256),
            _act(activation),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride, activation):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act = _act(activation)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm2d(out_c))
            if stride != 1 or in_c != out_c
            else nn.Identity()
        )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class MiniResNet(nn.Module):
    """Model C — Residual network. Better gradient flow on deeper stack. ~2.7M params."""

    def __init__(self, num_classes: int = NUM_CLASSES, activation: str = "relu", dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            _act(activation),
        )
        self.layer1 = nn.Sequential(ResidualBlock(64, 64, 1, activation), ResidualBlock(64, 64, 1, activation))
        self.layer2 = nn.Sequential(ResidualBlock(64, 128, 2, activation), ResidualBlock(128, 128, 1, activation))
        self.layer3 = nn.Sequential(ResidualBlock(128, 256, 2, activation), ResidualBlock(256, 256, 1, activation))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.head(x)


MODEL_REGISTRY = {
    "SimpleCNN":  {"cls": SimpleCNN,  "desc": "Shallow 2-conv baseline (~1.2M params). Trains fast, lower ceiling."},
    "DeeperCNN":  {"cls": DeeperCNN,  "desc": "VGG-style 6-conv with BatchNorm (~3.5M params). Good FER2013 accuracy."},
    "MiniResNet": {"cls": MiniResNet, "desc": "Residual network with skip connections (~2.8M params). Highest ceiling."},
}


def build_model(name: str, activation: str = "relu", dropout: float = 0.3, num_classes: int = NUM_CLASSES) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]["cls"](num_classes=num_classes, activation=activation, dropout=dropout)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
