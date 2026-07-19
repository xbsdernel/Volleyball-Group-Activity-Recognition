import torch


def compute_correct(outputs, labels):
    return (torch.argmax(outputs, 1) == labels).sum().item()
