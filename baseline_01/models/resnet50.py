import torch
import torch.nn as nn
from torchvision import models
from configs import config


def build_model(device=None, num_classes=config.NUM_CLASSES, use_channels_last=config.USE_CHANNELS_LAST):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(
        num_features,
        num_classes
    )
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    if device is not None:
        model = model.to(device)

    return model

