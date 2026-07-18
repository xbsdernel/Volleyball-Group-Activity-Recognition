import torch.nn as nn
from torchvision import models
from configs import config


def build_model(device):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(
        num_features,
        config.NUM_CLASSES
    )
    model = nn.DataParallel(model)
    model = model.to(device)

    return model
