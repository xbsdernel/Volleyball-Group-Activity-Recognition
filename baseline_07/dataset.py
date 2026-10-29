"""
Baseline 7: Dataset and Player Tracklet DataLoaders
Generates 12-player tracklets (12, 9, 3, 224, 224) for Baseline 7 evaluation and training.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

TRAIN_VIDEOS = {1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54}
VAL_VIDEOS   = {0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51}
TEST_VIDEOS  = {4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47}

CLASSES = ["l_pass", "r_pass", "l_spike", "r_spike", "l_set", "r_set", "l_winpoint", "r_winpoint"]
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
NUM_PLAYERS = 12
NUM_FRAMES = 9


class BoxInfo:
    def __init__(self, line: str):
        words = line.split()
        self.category = words.pop()
        words = [int(w) for w in words]
        self.player_ID = words[0]
        x1, y1, x2, y2, frame_ID, *_ = words[1:]
        self.box = (x1, y1, x2, y2)
        self.frame_ID = frame_ID


def load_tracking_annot(path: str):
    player_boxes = {i: [] for i in range(12)}
    frame_boxes_dct = {}
    if not os.path.isfile(path):
        return frame_boxes_dct
    with open(path, 'r') as f:
        for line in f:
            b = BoxInfo(line)
            if b.player_ID > 11:
                continue
            player_boxes[b.player_ID].append(b)
    for pid, boxes in player_boxes.items():
        boxes = boxes[5:-6] if len(boxes) > 11 else boxes
        for b in boxes:
            frame_boxes_dct.setdefault(b.frame_ID, []).append(b)
    return frame_boxes_dct


def get_seq_paths(mid_path: str, n: int = 4):
    d = os.path.dirname(mid_path)
    name, ext = os.path.splitext(os.path.basename(mid_path))
    mid = int(name)
    return [p if os.path.isfile(p := os.path.join(d, f"{mid + o}{ext}")) else mid_path for o in range(-n, n + 1)]


def load_b7_metadata(videos_root: str, tracking_root: str):
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

        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name, lbl = parts[0], parts[1]
                    clip_id = os.path.splitext(img_name)[0]
                    mid_path = os.path.join(vpath, clip_id, img_name)
                    trk_path = os.path.join(tracking_root, str(vid_id), clip_id, f"{clip_id}.txt")
                    data.append({
                        "video_id": vid_id,
                        "frame_paths": get_seq_paths(mid_path),
                        "tracking_path": trk_path,
                        "group_label": lbl,
                        "image_path": mid_path
                    })
    return data


def get_crop_transform(image_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class PersonCropDataset(Dataset):
    """Stage 1: Person action crop dataset."""
    def __init__(self, data, transform=None):
        self.samples = []
        self.transform = transform
        for d in data:
            if not os.path.isfile(d.get("image_path", "")) or not os.path.isfile(d["tracking_path"]):
                continue
            fboxes = load_tracking_annot(d["tracking_path"])
            if not fboxes:
                continue
            fid = list(fboxes.keys())[0]
            for b in fboxes[fid][:12]:
                self.samples.append((d["image_path"], b.box, PERSON_ACTION_MAP.get(b.category.lower(), 8)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        p, (x1, y1, x2, y2), act = self.samples[idx]
        img = Image.open(p).convert("RGB")
        w, h = img.size
        c = img.crop((max(0, x1), max(0, y1), min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))))
        if self.transform:
            c = self.transform(c)
        return c, torch.tensor(act, dtype=torch.long)


class HierarchicalDataset(Dataset):
    """Stage 2: 12-player tracklets dataset: produces (12, 9, 3, 224, 224) per sample."""
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        d = self.data[idx]
        fboxes = load_tracking_annot(d["tracking_path"])
        all_players = [[] for _ in range(NUM_PLAYERS)]
        for p in d["frame_paths"]:
            img = Image.open(p).convert("RGB")
            w, h = img.size
            fid_str = os.path.splitext(os.path.basename(p))[0]
            try:
                fid = int(fid_str)
            except ValueError:
                fid = list(fboxes.keys())[0] if fboxes else 0
            boxes = fboxes.get(fid, [])
            for k in range(NUM_PLAYERS):
                if k < len(boxes):
                    x1, y1, x2, y2 = boxes[k].box
                    crop = img.crop((max(0, x1), max(0, y1), min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))))
                else:
                    crop = Image.new("RGB", IMAGE_SIZE, 0)
                if self.transform:
                    crop = self.transform(crop)
                all_players[k].append(crop)

        player_tracklets = torch.stack([torch.stack(trk) for trk in all_players])  # (12, 9, 3, 224, 224)
        label = torch.tensor(LABEL_MAP[d["group_label"]], dtype=torch.long)
        return player_tracklets, label


def build_b7_dataloaders(videos_root: str, tracking_root: str, batch_size: int = 8, num_workers: int = 2):
    data = load_b7_metadata(videos_root, tracking_root)
    train_data = [d for d in data if d["video_id"] in TRAIN_VIDEOS]
    val_data   = [d for d in data if d["video_id"] in VAL_VIDEOS]
    test_data  = [d for d in data if d["video_id"] in TEST_VIDEOS]

    tf = get_crop_transform(IMAGE_SIZE)

    train_ds = HierarchicalDataset(train_data, transform=tf)
    val_ds   = HierarchicalDataset(val_data,   transform=tf)
    test_ds  = HierarchicalDataset(test_data,  transform=tf)

    loader_kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=(num_workers > 0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader
