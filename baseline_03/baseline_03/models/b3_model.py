import torch
import torch.nn as nn
from torchvision import models

from configs import config


class B3Model(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.classifier = nn.Linear(2048, config.NUM_CLASSES)

    def forward(self, x):
        B, P, C, H, W = x.shape
        x = x.view(B * P, C, H, W)
        features = self.backbone(x)
        features = features.view(B, P, 2048)
        features = features.mean(dim=1)
        out = self.classifier(features)

        return out


def build_model(device):
    model = B3Model()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)

    return model
