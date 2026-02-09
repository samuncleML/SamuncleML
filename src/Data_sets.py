from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import PIL.Image as Image
import torch
import pandas as pd
import numpy as np


transform = transforms.Compose([
    transforms.RandomAdjustSharpness(1.5),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop((224, 224), scale=(0.2, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.42636684, 0.48088259, 0.34474218), (0.18698326, 0.18891238, 0.18710552))
    ])

tt = torch.tensor

class DropDiseaseDataset(Dataset):
    def __init__(self, csv_file: str, transform=None):
        super().__init__()
        df = pd.read_csv(csv_file)
        self.data = df.to_numpy()
        self.transform = transform
        self.disease_to_idx = {name: i for i, name in enumerate(np.unique(self.data[:, 1]))}
        self.plant_to_idx = {name: i for i, name in enumerate(np.unique(self.data[:, 0]))}
        print(self.disease_to_idx.keys())
        print(self.plant_to_idx.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        image_path = self.data[index, 2]
        image = Image.open(image_path).convert('RGB')
        crop_type = self.plant_to_idx[self.data[index, 0]]
        disease_type = self.disease_to_idx[self.data[index, 1]]
        if image:
            image = self.transform(image)
        return image, tt(crop_type), tt(disease_type)

leaves = DropDiseaseDataset('./crop_disease_labels.csv', transform=transform)
len_train = int(0.25*len(leaves))
len_val = len(leaves) - len_train
train_data, val_data = random_split(leaves, lengths=[len_train, len_val])
train_loader = DataLoader(train_data, batch_size=30, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=20)