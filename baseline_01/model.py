"""
Baseline 1: Image Classification Model
Paper: Ibrahim et al., CVPR 2016 (Single middle frame classification)
Backbone: ResNet-50 (replacing AlexNet per agreed substitution)
Head: Dropout(0.3) -> Linear(2048, 8)
"""

import torch
import torch.nn as nn
from torchvision import models


class B1ImageClassifier(nn.Module):
    """
    Baseline 1: Image Classification.
    Extracts features from the central frame of each clip using a ResNet-50 backbone,
    followed by a dropout regularized linear classification head.
    """
    def __init__(self, num_classes: int = 8, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_feature_extractor(self) -> nn.Module:
        """Returns the model with classification head replaced by Identity for feature extraction."""
        self.backbone.fc = nn.Identity()
        return self.backbone
