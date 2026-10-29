"""
Baseline 4: Dataset and Sequence DataLoaders
Loads 9-frame sequences (middle frame +/- 4) for temporal modeling with image features.
"""

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder

CLASSES = [
    'r_set', 'r_spike', 'r-pass', 'r_winpoint',
    'l_set', 'l-spike', 'l-pass', 'l_winpoint'
]

TRAIN_VIDEOS = {1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54}
VAL_VIDEOS   = {0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51}
TEST_VIDEOS  = {4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47}

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

_clip_spatial = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
])

_clip_color = transforms.ColorJitter(
    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
)

_to_tensor_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def get_sequence_frame_paths(middle_frame_path: str, num_frames_around: int = 4):
    """Returns 2*num_frames_around + 1 frame paths centered on middle_frame_path."""
    clip_folder_path = os.path.dirname(middle_frame_path)
    middle_frame_num = int(os.path.splitext(os.path.basename(middle_frame_path))[0])

    sequence_paths = []
    for i in range(-num_frames_around, num_frames_around + 1):
        frame_path = os.path.join(clip_folder_path, f"{middle_frame_num + i}.jpg")
        if not os.path.isfile(frame_path):
            return None
        sequence_paths.append(frame_path)
    return sequence_paths


def load_sequence_data(data_root: str, num_frames_around: int = 4):
    """Loads all valid 9-frame sequence annotations from the dataset."""
    data = []
    if not os.path.isdir(data_root):
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
                    seq = get_sequence_frame_paths(image_path, num_frames_around=num_frames_around)
                    if seq is not None:
                        data.append((vid_id, seq, activity))
    return data


class CustomSequenceDataset(Dataset):
    """Dataset for fixed-length video frame sequences."""
    def __init__(self, data, label_encoder: LabelEncoder, is_train: bool = False):
        self.sequences = [item[0] for item in data]
        self.is_train = is_train
        raw_labels = [item[1] for item in data]
        self.labels = torch.as_tensor(label_encoder.transform(raw_labels), dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence_paths = self.sequences[idx]
        frames = []

        if self.is_train:
            clip_seed = torch.randint(0, 2**31, (1,)).item()
            for img_path in sequence_paths:
                img = Image.open(img_path).convert('RGB')
                torch.manual_seed(clip_seed)
                random.seed(clip_seed)
                img = _clip_spatial(img)
                img = _clip_color(img)
                img = _to_tensor_norm(img)
                frames.append(img)
        else:
            for img_path in sequence_paths:
                img = Image.open(img_path).convert('RGB')
                img = val_test_transform(img)
                frames.append(img)

        return torch.stack(frames), self.labels[idx]


def build_sequence_dataloaders(data_root: str, batch_size: int = 2, num_workers: int = 2):
    """Builds train, val, and test loaders for sequence data."""
    all_data = load_sequence_data(data_root, num_frames_around=4)

    train_data = [(seq, act) for v, seq, act in all_data if v in TRAIN_VIDEOS]
    val_data   = [(seq, act) for v, seq, act in all_data if v in VAL_VIDEOS]
    test_data  = [(seq, act) for v, seq, act in all_data if v in TEST_VIDEOS]

    label_encoder = LabelEncoder()
    label_encoder.fit(CLASSES)

    train_ds = CustomSequenceDataset(train_data, label_encoder, is_train=True)
    val_ds   = CustomSequenceDataset(val_data,   label_encoder, is_train=False)
    test_ds  = CustomSequenceDataset(test_data,  label_encoder, is_train=False)

    loader_kw = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader, label_encoder
