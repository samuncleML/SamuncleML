import pandas as pd
import torch
import PIL.Image as Image
from torchvision import transforms
import numpy as np
std_list = []
mean_list = []


data = pd.read_csv('data/train.csv')
for path in data['image']:
    image = Image.open(path).convert('RGB')
    image = image.resize((224, 224))
    image = torch.tensor(transforms.ToTensor()(image))
    std = image.std(dim=[1, 2])
    mean = image.mean(dim=[1, 2])

    std_list.append(std)
    mean_list.append(mean)

std_list = np.array(std_list)
mean_list = np.array(mean_list)

print(std_list.mean(axis=0))
print(mean_list.mean(axis=0))