"""
Baseline 5: Temporal Model with Person Features
Paper: Ibrahim et al., CVPR 2016 — Section 4.1, Baseline 5
Architecture:
- Off-the-shelf ImageNet CNN features (ResNet-50 backbone, frozen)
- Extracts features from 12 players per frame across 9 frames: (B, 9, 12, 2048)
- Max-pooling across 12 players at each timestep: Z_t = max_k (x_tk) -> (B, 9, 2048)
- Group LSTM (input_size=2048, hidden_size=500, num_layers=1)
- Dropout(0.5) -> Linear(500, 8)
"""

import torch
import torch.nn as nn
from torchvision import models


class B5Model(nn.Module):
    """
    Baseline 5: Off-the-shelf person features with temporal LSTM.
    """
    def __init__(self, num_classes: int = 8, hidden_dim: int = 500, dropout: float = 0.5, pretrained_backbone: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.backbone.parameters():
            p.requires_grad = False  # Off-the-shelf ImageNet CNN features frozen

        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 9, 12, 3, 224, 224)
        B, T, K, C, H, W = x.shape
        x_flat = x.view(B * T * K, C, H, W)
        with torch.no_grad():
            feats = self.backbone(x_flat).view(B, T, K, 2048)

        # Max-pool across 12 players per frame: Z_t = max_k (x_tk) [Eq. 8]
        scene_seq = feats.max(dim=2)[0]  # (B, 9, 2048)
        _, (hn, _) = self.lstm(scene_seq)
        return self.classifier(self.dropout(hn[-1]))
