import torchvision.transforms as transforms
from configs import config

def get_transforms():
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.MEAN,
            std=config.STD
        )
    ])
    return transform
