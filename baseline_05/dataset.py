"""
Baseline 5: Dataset and Tracking Annotation Parsers
Loads 9-frame sequences with 12 player crops per frame: (9, 12, 3, 224, 224).
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

IMAGE_SIZE = (224, 224)


class BoxInfo:
    def __init__(self, line: str):
        words = line.split()
        self.category = words.pop()
        words = [int(w) for w in words]
        self.player_ID = words[0]
        x1, y1, x2, y2, frame_ID, lost, grouping, generated = words[1:]
        self.box = (x1, y1, x2, y2)
        self.frame_ID = frame_ID


def load_tracking_annot(path: str):
    player_boxes = {idx: [] for idx in range(12)}
    frame_boxes_dct = {}
    if not os.path.isfile(path):
        return frame_boxes_dct
    with open(path, 'r') as file:
        for line in file:
            b = BoxInfo(line)
            if b.player_ID > 11:
                continue
            player_boxes[b.player_ID].append(b)
    for player_ID, boxes_info in player_boxes.items():
        boxes_info = boxes_info[5:-6] if len(boxes_info) > 11 else boxes_info
        for b in boxes_info:
            if b.frame_ID not in frame_boxes_dct:
                frame_boxes_dct[b.frame_ID] = []
            frame_boxes_dct[b.frame_ID].append(b)
    return frame_boxes_dct


def get_sequence_frame_paths(middle_frame_path: str, num_frames_around: int = 4):
    clip_dir = os.path.dirname(middle_frame_path)
    base_name = os.path.basename(middle_frame_path)
    name, ext = os.path.splitext(base_name)
    mid_id = int(name)
    paths = []
    for offset in range(-num_frames_around, num_frames_around + 1):
        p = os.path.join(clip_dir, f"{mid_id + offset}{ext}")
        paths.append(p if os.path.isfile(p) else middle_frame_path)
    return paths


def load_b5_metadata(videos_root: str, tracking_root: str, num_frames_around: int = 4):
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
                    img_name, grp_lbl = parts[0], parts[1]
                    clip_id = os.path.splitext(img_name)[0]
                    mid_img_path = os.path.join(vpath, clip_id, img_name)
                    trk_path = os.path.join(tracking_root, str(vid_id), clip_id, f"{clip_id}.txt")
                    seq_paths = get_sequence_frame_paths(mid_img_path, num_frames_around=num_frames_around)
                    data.append({
                        "video_id": vid_id,
                        "clip_id": clip_id,
                        "frame_paths": seq_paths,
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


class VolleyballPersonSeqDataset(Dataset):
    """Generates (9, 12, 3, 224, 224) player crops tensor per clip."""
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        d = self.data[idx]
        fboxes = load_tracking_annot(d["tracking_path"])
        all_frames = []
        for p in d["frame_paths"]:
            img = Image.open(p).convert("RGB")
            fid_str = os.path.splitext(os.path.basename(p))[0]
            try:
                fid = int(fid_str)
            except ValueError:
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
            all_frames.append(torch.stack(crops))
        clip = torch.stack(all_frames)  # (9, 12, 3, 224, 224)
        label = torch.tensor(LABEL_MAP[d["group_label"]], dtype=torch.long)
        return clip, label


def build_b5_dataloaders(videos_root: str, tracking_root: str, batch_size: int = 16, num_workers: int = 2):
    data = load_b5_metadata(videos_root, tracking_root)
    train_data = [d for d in data if d["video_id"] in TRAIN_VIDEOS]
    val_data   = [d for d in data if d["video_id"] in VAL_VIDEOS]
    test_data  = [d for d in data if d["video_id"] in TEST_VIDEOS]

    tf = get_crop_transform(IMAGE_SIZE)

    train_ds = VolleyballPersonSeqDataset(train_data, transform=tf)
    val_ds   = VolleyballPersonSeqDataset(val_data,   transform=tf)
    test_ds  = VolleyballPersonSeqDataset(test_data,  transform=tf)

    loader_kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=(num_workers > 0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader
