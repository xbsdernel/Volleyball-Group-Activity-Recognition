from torch.utils.data import DataLoader

from configs import config
from datasets.custom_dataset import VolleyballDataset
from datasets.transforms import get_transforms


def get_datasets(train_data, val_data, test_data):
    transform = get_transforms()

    train_dataset = VolleyballDataset(train_data, transform)
    val_dataset = VolleyballDataset(val_data, transform)
    test_dataset = VolleyballDataset(test_data, transform)

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
