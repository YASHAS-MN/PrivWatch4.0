# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
# pylint: disable=import-error
"""MobileNetV2 + LSTM Video Activity Recognition Model."""

import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class ActionModel(nn.Module):
    """Activity recognition model combining MobileNetV2 spatial backbone with an LSTM temporal head.

    Architecture:
        Input: (B, T, C, H, W)
        1. Reshape to (B * T, C, H, W)
        2. MobileNetV2 features -> (B * T, 1280, H', W')
        3. AdaptiveAvgPool2d(1) -> (B * T, 1280, 1, 1)
        4. Reshape to (B, T, 1280)
        5. LSTM(1280 -> 128, batch_first=True) -> (B, T, 128)
        6. Select final timestep -> (B, 128)
        7. Linear(128 -> 3) -> 3 class logits: Normal (0), Fight (1), Collapse (2)
    """

    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained_backbone else None
        backbone = mobilenet_v2(weights=weights)
        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(1280, 128, batch_first=True)
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size * seq_len, channels, height, width)
        x = self.cnn(x)
        x = self.pool(x)
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)

