import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from PIL import Image

label_encoder = LabelEncoder()


class CustomImageDataset(Dataset):
    def __init__(self, data, transform=None, label_encoder_instance=None):
        self.data = data
        self.transform = transform
        encoder = label_encoder_instance if label_encoder_instance is not None else label_encoder
        if not hasattr(encoder, "classes_") or len(encoder.classes_) == 0:
            labels_raw = [item[1] for item in data]
            self.labels = torch.tensor(encoder.fit_transform(labels_raw), dtype=torch.long)
        else:
            labels_raw = [item[1] for item in data]
            self.labels = torch.tensor(encoder.transform(labels_raw), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx][0]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

