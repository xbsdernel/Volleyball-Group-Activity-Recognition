"""
Baseline 3: Dataset and Tracking Annotation Parsers
Handles bounding box tracklets for 12 players per frame.
Stage 1: SinglePlayerCropDataset for person action fine-tuning.
Stage 2: Volleyball12PlayersDataset for max-pooling group activity recognition.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

TRAIN_VIDEOS = {1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54}
VAL_VIDEOS   = {0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51}
TEST_VIDEOS  = {4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47}

CLASSES = ['l_pass', 'r_pass', 'l_spike', 'r_spike', 'l_set', 'r_set', 'l_winpoint', 'r_winpoint']

LABEL_MAP = {
    "l_pass": 0, "r_pass": 1, "l_spike": 2, "r_spike": 3,
    "l_set": 4, "r_set": 5, "l_winpoint": 6, "r_winpoint": 7,
    "l-pass": 0, "r-pass": 1, "l-spike": 2, "r-spike": 3,
    "l-set": 4, "r-set": 5, "l-winpoint": 6, "r-winpoint": 7
}

PERSON_ACTION_MAP = {
    "waiting": 0, "setting": 1, "digging": 2, "falling": 3,
    "spiking": 4, "blocking": 5, "jumping": 6, "moving": 7, "standing": 8
}

IMAGE_SIZE = (224, 224)


class BoxInfo:
    """Parses a single line of volleyball tracking annotation."""
    def __init__(self, line: str):
        words = line.split()
        self.category = words.pop()
        words = [int(w) for w in words]
        self.player_ID = words[0]
        x1, y1, x2, y2, frame_ID, lost, grouping, generated = words[1:]
        self.box = (x1, y1, x2, y2)
        self.frame_ID = frame_ID


def load_tracking_annot(path: str):
    """Loads player tracking annotations indexed by frame ID."""
    player_boxes = {idx: [] for idx in range(12)}
    frame_boxes_dct = {}
    if not os.path.isfile(path):
        return frame_boxes_dct
    with open(path, 'r') as file:
        for line in file:
            box_info = BoxInfo(line)
            if box_info.player_ID > 11:
                continue
            player_boxes[box_info.player_ID].append(box_info)

    for player_ID, boxes_info in player_boxes.items():
        boxes_info = boxes_info[5:-6] if len(boxes_info) > 11 else boxes_info
        for box_info in boxes_info:
            if box_info.frame_ID not in frame_boxes_dct:
                frame_boxes_dct[box_info.frame_ID] = []
            frame_boxes_dct[box_info.frame_ID].append(box_info)
    return frame_boxes_dct


def load_volleyball_metadata(videos_root: str, tracking_root: str):
    """Scans all videos and pairs frame paths with tracking annotations."""
    data = []
    if not os.path.isdir(videos_root):
        return data

    for vid_name in sorted(os.listdir(videos_root)):
        vpath = os.path.join(videos_root, vid_name)
        if not os.path.isdir(vpath):
            continue
        ann_path = os.path.join(vpath, "annotations.txt")
        if not os.path.isfile(ann_path):
            continue
        try:
            vid_id = int(vid_name)
        except ValueError:
            continue

        with open(ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    grp_lbl = parts[1]
                    clip_id = os.path.splitext(img_name)[0]
                    img_path = os.path.join(vpath, clip_id, img_name)
                    trk_path = os.path.join(tracking_root, str(vid_id), clip_id, f"{clip_id}.txt")
                    data.append({
                        "video_id": vid_id,
                        "clip_id": clip_id,
                        "image_path": img_path,
                        "tracking_path": trk_path,
                        "group_label": grp_lbl
                    })
    return data


def get_crop_transform(image_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class SinglePlayerCropDataset(Dataset):
    """Stage 1: Extracts individual player crops for person action fine-tuning."""
    def __init__(self, data, transform=None):
        self.samples = []
        self.transform = transform
        for d in data:
            if not os.path.isfile(d["image_path"]) or not os.path.isfile(d["tracking_path"]):
                continue
            fboxes = load_tracking_annot(d["tracking_path"])
            if not fboxes:
                continue
            fid = list(fboxes.keys())[0]
            for b in fboxes[fid][:12]:
                act_idx = PERSON_ACTION_MAP.get(b.category.lower(), 8)
                self.samples.append((d["image_path"], b.box, act_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, (x1, y1, x2, y2), act_idx = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        crop = img.crop((max(0, x1), max(0, y1), min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))))
        if self.transform:
            crop = self.transform(crop)
        return crop, torch.tensor(act_idx, dtype=torch.long)


class Volleyball12PlayersDataset(Dataset):
    """Stage 2: Extracts crops for all 12 players in the middle frame."""
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        d = self.data[idx]
        img = Image.open(d["image_path"]).convert("RGB")
        fboxes = load_tracking_annot(d["tracking_path"])
        fid = list(fboxes.keys())[0] if fboxes else 0
        boxes = fboxes.get(fid, [])
        crops = []
        w, h = img.size
        for b in boxes[:12]:
            x1, y1, x2, y2 = b.box
            c = img.crop((max(0, x1), max(0, y1), min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))))
            if self.transform:
                c = self.transform(c)
            crops.append(c)
        while len(crops) < 12:
            crops.append(torch.zeros(3, *IMAGE_SIZE))
        return torch.stack(crops), torch.tensor(LABEL_MAP[d["group_label"]], dtype=torch.long)


def build_b3_dataloaders(videos_root: str, tracking_root: str, batch_size: int = 16, num_workers: int = 2):
    """Constructs train, val, and test DataLoaders for Stage 2 B3Model."""
    data = load_volleyball_metadata(videos_root, tracking_root)
    train_data = [d for d in data if d["video_id"] in TRAIN_VIDEOS]
    val_data   = [d for d in data if d["video_id"] in VAL_VIDEOS]
    test_data  = [d for d in data if d["video_id"] in TEST_VIDEOS]

    tf = get_crop_transform(IMAGE_SIZE)

    train_ds = Volleyball12PlayersDataset(train_data, transform=tf)
    val_ds   = Volleyball12PlayersDataset(val_data,   transform=tf)
    test_ds  = Volleyball12PlayersDataset(test_data,  transform=tf)

    loader_kw = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader
