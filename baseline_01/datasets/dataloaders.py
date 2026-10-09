from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from configs import config
from datasets.custom_dataset import CustomImageDataset, label_encoder
from datasets.transforms import get_transforms


def get_datasets(train_data, val_data, test_data):
    transform = get_transforms()

    # Fit label_encoder ONCE across all dataset splits to guarantee consistent class indexing
    all_activities = sorted(list(set([item[1] for item in train_data + val_data + test_data])))
    label_encoder.fit(all_activities)

    train_dataset = CustomImageDataset(data=train_data, transform=transform, label_encoder_instance=label_encoder)
    val_dataset = CustomImageDataset(data=val_data, transform=transform, label_encoder_instance=label_encoder)
    test_dataset = CustomImageDataset(data=test_data, transform=transform, label_encoder_instance=label_encoder)

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(train_dataset, val_dataset, test_dataset,
                    batch_size=config.BATCH_SIZE,
                    num_workers=config.NUM_WORKERS,
                    is_distributed=False,
                    rank=0,
                    world_size=1):

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if is_distributed else None

    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    ) if is_distributed else None

    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    ) if is_distributed else None

    use_persistent = config.PERSISTENT_WORKERS and num_workers > 0
    prefetch = config.PREFETCH_FACTOR if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch
    )

    return train_loader, val_loader, test_loader, train_sampler

