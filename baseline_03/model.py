"""
Baseline 3: Fine-tuned Person Classification Model
Paper: Ibrahim et al., CVPR 2016 — Section 3.2, 4.1, 4.3
Stage 1: Person Action CNN (ResNet-50 backbone -> Linear(2048, 9))
Stage 2: B3Model with MAX-POOLING over all 12 players: Z = max_k(x_k) -> Linear(2048, 8)
"""

import torch
import torch.nn as nn
from torchvision import models


class PersonActionCNN(nn.Module):
    """
    Stage 1: Person Action Classifier.
    Fine-tunes the CNN backbone on individual cropped player bounding boxes
    with 9 atomic action categories.
    """
    def __init__(self, num_actions: int = 9, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Linear(2048, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x).view(x.size(0), -1)
        return self.classifier(feats)


class B3Model(nn.Module):
    """
    Stage 2: Baseline 3 Group Activity Model.
    Uses the fine-tuned person CNN backbone (frozen) to extract features from all 12 players,
    applies MAX-POOLING across players: Z = max_k(x_k), followed by a linear group classifier.
    """
    def __init__(self, fine_tuned_backbone: nn.Module = None, num_classes: int = 8, freeze_backbone: bool = True):
        super().__init__()
        if fine_tuned_backbone is not None:
            self.backbone = fine_tuned_backbone
        else:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False  # Backbone frozen in Stage 2 as in paper

        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 12, 3, 224, 224)
        B, K, C, H, W = x.shape
        x_flat = x.view(B * K, C, H, W)
        with torch.no_grad():
            feats = self.backbone(x_flat).view(B, K, 2048)

        # MAX-POOLING over all 12 players: Z = max_k (x_k) [Paper Section 3.2, 4.1, 4.3]
        z = feats.max(dim=1)[0]  # (B, 2048)
        return self.classifier(z)
