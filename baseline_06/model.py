"""
Baseline 6: Two-Stage Model Without LSTM 1
Paper: Ibrahim et al., CVPR 2016 — Section 4.1, Baseline 6
Stage 1: Fine-tuned Person Action CNN (ResNet-50 backbone -> Linear(2048, 9))
Stage 2: B6Model:
- Fine-tuned person CNN backbone (frozen)
- Max-pool over 12 players per frame: Z_t = max_k (x_tk) -> (B, 9, 2048)
- Group LSTM 2 (input_size=2048, hidden_size=500, num_layers=1)
- Dropout(0.5) -> Linear(500, 8)
"""

import torch
import torch.nn as nn
from torchvision import models


class PersonActionCNN(nn.Module):
    """
    Stage 1: Person Action Classifier.
    Fine-tunes the CNN backbone on individual cropped player bounding boxes.
    """
    def __init__(self, num_actions: int = 9, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Linear(2048, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x).view(x.size(0), -1)
        return self.classifier(f)


class B6Model(nn.Module):
    """
    Stage 2: Baseline 6 Group Activity Recognition (Two-stage model WITHOUT LSTM 1).
    Frozen fine-tuned person CNN -> max-pool over players per frame -> Group LSTM 2.
    """
    def __init__(
        self,
        finetuned_backbone: nn.Module = None,
        num_classes: int = 8,
        hidden_dim: int = 500,
        dropout: float = 0.5,
        freeze_backbone: bool = True
    ):
        super().__init__()
        if finetuned_backbone is not None:
            self.backbone = finetuned_backbone
        else:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False  # Frozen fine-tuned person CNN

        # Group LSTM 2 only (no person-level LSTM 1)
        self.lstm2 = nn.LSTM(input_size=2048, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 9, 12, 3, 224, 224)
        B, T, K, C, H, W = x.shape
        x_flat = x.view(B * T * K, C, H, W)
        with torch.no_grad():
            feats = self.backbone(x_flat).view(B, T, K, 2048)

        # Max-pool over players per frame
        scene_seq = feats.max(dim=2)[0]  # (B, 9, 2048)

        # Group LSTM 2
        _, (hn, _) = self.lstm2(scene_seq)
        return self.classifier(self.dropout(hn[-1]))
