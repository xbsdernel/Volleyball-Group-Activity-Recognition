import torch
from torch.utils.data import Dataset
from PIL import Image

from configs import config
from datasets.tracking import load_tracking_annot


class VolleyballDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        image = Image.open(sample["image_path"]).convert("RGB")

        frame_boxes = load_tracking_annot(sample["tracking_path"])

        frame_id = list(frame_boxes.keys())[0]

        player_crops = []

        for box_info in frame_boxes[frame_id][:config.NUM_PLAYERS]:
            x1, y1, x2, y2 = box_info.box

            crop = image.crop((x1, y1, x2, y2))

            if self.transform:
                crop = self.transform(crop)

            player_crops.append(crop)

        while len(player_crops) < config.NUM_PLAYERS:
            player_crops.append(torch.zeros(3, *config.IMAGE_SIZE))

        player_crops = torch.stack(player_crops)

        label = torch.tensor(
            config.LABEL_MAP[sample["group_label"]],
            dtype=torch.long
        )

        return player_crops, label
