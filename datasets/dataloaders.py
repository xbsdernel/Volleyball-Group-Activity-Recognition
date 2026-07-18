from torch.utils.data import DataLoader

from configs import config
from datasets.custom_dataset import CustomImageDataset
from datasets.transforms import get_transforms


def get_datasets(train_data, val_data, test_data):
    transform = get_transforms()
    train_dataset = CustomImageDataset(data=train_data, transform=transform)
    val_dataset = CustomImageDataset(data=val_data, transform=transform)
    test_dataset = CustomImageDataset(data=test_data, transform=transform)

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    return train_loader, val_loader, test_loader
