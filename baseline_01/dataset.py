"""
Baseline 1: Dataset and DataLoaders
Loads annotated middle-frame images for single-frame activity recognition.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder

CLASSES = [
    'r_set', 'r_spike', 'r-pass', 'r_winpoint',
    'l_set', 'l-spike', 'l-pass', 'l_winpoint'
]

# Official 55-Video Splits (avoids video-level leakage)
TRAIN_VIDEOS = {1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54}
VAL_VIDEOS   = {0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51}
TEST_VIDEOS  = {4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47}


def get_transforms(image_size=(224, 224)):
    """Returns training and validation/test image transformations."""
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_test_transform


def load_dataset_annotations(data_root: str):
    """Parses annotations.txt files from all video subdirectories."""
    data = []
    if not os.path.exists(data_root):
        return data

    for video_folder_name in sorted(os.listdir(data_root)):
        current_video_folder_path = os.path.join(data_root, video_folder_name)
        if not os.path.isdir(current_video_folder_path):
            continue

        annotations_path = os.path.join(current_video_folder_path, "annotations.txt")
        if not os.path.isfile(annotations_path):
            continue

        try:
            vid_id = int(video_folder_name)
        except ValueError:
            continue

        with open(annotations_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_filename = parts[0]
                    activity = parts[1]
                    name_folder = image_filename.split('.')[0]
                    image_path = os.path.join(current_video_folder_path, name_folder, image_filename)
                    data.append((vid_id, image_path, activity))
    return data


def get_splits(data):
    """Splits samples into train, validation, and test sets based on video IDs."""
    train_data = [(p, a) for v, p, a in data if v in TRAIN_VIDEOS]
    val_data   = [(p, a) for v, p, a in data if v in VAL_VIDEOS]
    test_data  = [(p, a) for v, p, a in data if v in TEST_VIDEOS]
    return train_data, val_data, test_data


class CustomImageDataset(Dataset):
    """Dataset for loading single middle-frame volleyball activity images."""
    def __init__(self, data, label_encoder: LabelEncoder, transform=None):
        self.image_paths = [item[0] for item in data]
        self.transform = transform
        raw_labels = [item[1] for item in data]
        self.labels = torch.as_tensor(label_encoder.transform(raw_labels), dtype=torch.long)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def build_dataloaders(data_root: str, batch_size: int = 64, num_workers: int = 2, image_size=(224, 224)):
    """Convenience function to build train, val, and test DataLoaders."""
    data = load_dataset_annotations(data_root)
    train_data, val_data, test_data = get_splits(data)

    label_encoder = LabelEncoder()
    label_encoder.fit(CLASSES)

    train_tf, val_tf = get_transforms(image_size)

    train_ds = CustomImageDataset(train_data, label_encoder, transform=train_tf)
    val_ds   = CustomImageDataset(val_data,   label_encoder, transform=val_tf)
    test_ds  = CustomImageDataset(test_data,  label_encoder, transform=val_tf)

    loader_kw = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader, label_encoder
