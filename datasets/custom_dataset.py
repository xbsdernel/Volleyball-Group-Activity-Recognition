import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from PIL import Image

label_encoder = LabelEncoder()
class CustomImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.labels = torch.tensor(label_encoder.fit_transform([item[1] for item in data]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx][0]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = (self.transform(image))

        return image, label
