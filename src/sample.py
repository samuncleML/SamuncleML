import torch

model = torch.load(r'C:\Users\Administrator\Documents\plant-disease-multihead\models\mobilenet_v3_large_latest.pt', map_location=torch.device('cpu'))
print(model)


