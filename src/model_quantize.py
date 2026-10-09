from Module import PlantDiseaseModel
import onnxscript
import torch
from PIL import Image
from torchvision import transforms
from torch import nn
from matplotlib import pyplot as plt

model = PlantDiseaseModel(5, 29)
checkpoint = torch.load(r'C:\Users\Administrator\Documents\plant-disease-multihead\models\dual_head_mobilenetv3_small.pt', map_location=torch.device('cpu'))
weights    = checkpoint['model_state_dict']
model.load_state_dict(weights)

new_model = nn.Sequential(
    model.shared,
    model.disease_head
)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize((0.426, 0.481, 0.344), (0.182, 0.184, 0.181))
    ])

image = Image.open(r"C:\Users\Administrator\Pictures\Screenshots\Screenshot 2026-06-23 132426.png").convert("RGB")
image = transform(image)

out = model(image.unsqueeze(0))
print(out)
