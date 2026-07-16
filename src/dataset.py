from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import PIL.Image as Image
import torch
import pandas as pd
import numpy as np
import os

BASE_PATH = os.getcwd()

transform = transforms.Compose([
    transforms.RandomAdjustSharpness(1.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.426, 0.481, 0.344), (0.182, 0.184, 0.181))
    ])

class CropDiseaseDataset(Dataset):
    def __init__(self, csv_file: str, transform=None):
        super().__init__()
        df = pd.read_csv(csv_file)
        self.data = df.to_numpy()
        self.transform = transform
        self.disease_to_idx = {name: i for i, name in enumerate(np.unique(self.data[:, 1]))}
        self.plant_to_idx = {name: i for i, name in enumerate(np.unique(self.data[:, 0]))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        image_path = self.data[index, 2]
        image = Image.open(image_path).convert('RGB')
        crop_type = self.plant_to_idx[self.data[index, 0]]
        disease_type = self.disease_to_idx[self.data[index, 1]]
        if image:
            image = self.transform(image)
        return image, torch.tensor(crop_type), torch.tensor(disease_type)


train_data = CropDiseaseDataset(os.path.join(BASE_PATH, 'data/train.csv'), transform=transform)
test_data = CropDiseaseDataset(os.path.join(BASE_PATH, 'data/test.csv'), transform=transform)
val_data = CropDiseaseDataset(os.path.join(BASE_PATH, 'data/val.csv'), transform=transform)

train_loader = DataLoader(train_data, batch_size=30, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=32)
val_loader = DataLoader(val_data, batch_size=32)
