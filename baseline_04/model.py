"""
Baseline 4: Temporal Model with Image Features
Paper: Ibrahim et al., CVPR 2016 — Section 4.1, Baseline 4
Architecture:
- ResNet-50 CNN Backbone (extracts 2048-dim feature representation per frame)
- Sequence of T=9 frames (middle frame +/- 4)
- Stacked LSTM (input_size=2048, hidden_size=256 or 500, num_layers=2, dropout=0.5)
- Linear classification head -> 8 group activity classes
"""

import torch
import torch.nn as nn
from torchvision import models


class LSTMClassifier(nn.Module):
    """
    Temporal LSTM classifier operating on precomputed or pooled sequence features.
    """
    def __init__(
        self,
        input_size: int = 2048,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 8,
        dropout_prob: float = 0.5
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_size)
        _, (hn, _) = self.lstm(x)     # hn: (num_layers, B, hidden_size)
        out = self.dropout(hn[-1])    # last-layer hidden state -> (B, hidden_size)
        return self.fc(out)


class B4Model(nn.Module):
    """
    End-to-End Baseline 4 Model.
    Processes full image frame sequences (B, T, 3, H, W) through ResNet-50 then LSTM.
    """
    def __init__(
        self,
        num_classes: int = 8,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        pretrained_backbone: bool = True
    ):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm_classifier = LSTMClassifier(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout_prob=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats = self.backbone(x_flat).view(B, T, 2048)
        return self.lstm_classifier(feats)
