import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from Module import PlantDiseaseModel
from torchmetrics.classification import MulticlassF1Score
from torchmetrics import Accuracy
from Data_sets import train_loader, val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = PlantDiseaseModel(5, 17)
model = model.to(device)
model.to(device)

epochs = 80
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
f1score_plant = MulticlassF1Score(num_classes=5, average="macro").to(device)
f1score_disease = MulticlassF1Score(num_classes=17, average="macro").to(device)
accuracy_plant = Accuracy(task='multiclass', num_classes=5).to(device)
accuracy_disease = Accuracy(task='multiclass', num_classes=17).to(device)

model.train()
for epoch in range(1, epochs+1):
    running_loss = 0.0
    for image, crop, disease in train_loader:
        optimizer.zero_grad()
        image = image.to(device)
        crop = crop.to(device)
        disease = disease.to(device)

        plant_outputs, disease_outputs = model.forward(image)
        plant_loss = criterion(plant_outputs, crop)
        disease_loss = criterion(disease_outputs, disease)
        f1score_plant.update(plant_outputs, crop)
        accuracy_plant.update(plant_outputs, crop)

        f1score_disease.update(disease_outputs, disease)
        accuracy_disease.update(disease_outputs, disease)
        combined_loss = plant_loss+disease_loss
        combined_loss.backward()

        running_loss += combined_loss.item()
        optimizer.step()

    f1_plant = f1score_plant.compute().item()
    acc_plant = accuracy_plant.compute().item()
    f1_disease = f1score_disease.compute().item()
    acc_disease = accuracy_disease.compute().item()
    print(f'Epoch--{epoch} Loss-- {running_loss/len(train_loader):.2f} -- Plant F1 Score-- {f1_plant*100:.2f}% -- Disease F1 Score-- {f1_disease*100:.2f}%  -- Plant Accuracy-- {acc_plant*100:.2f}% -- Disease Accuracy-- {acc_disease*100:.2f}%')
    
    f1score_plant.reset()
    f1score_disease.reset()
    f1score_disease.reset()
    accuracy_plant.reset()

model.eval()
with torch.no_grad():
    for image, crop, disease in val_loader:
        optimizer.zero_grad()
        image = image.to(device)
        crop = crop.to(device)
        disease = disease.to(device)

        plant_outputs, disease_outputs = model(image)
        f1score_plant.update(plant_outputs, crop)
        accuracy_plant.update(plant_outputs, crop)

        f1score_disease.update(disease_outputs, disease)
        accuracy_disease.update(disease_outputs, disease)
        running_loss += combined_loss.item()
        optimizer.step()
    acc_plant = accuracy_plant.compute().item()
    acc_disease = accuracy_disease.compute().item()
    f1_plant = f1score_plant.compute().item()
    f1_disease = f1score_disease.compute().item()
    print('------------- VALIDATION -------------')
    print(f'Plant F1 Score-- {f1_plant*100:.2f}% -- Disease Accuracy-- {f1_disease*100:.2f}% --  Plant Accuracy-- {acc_plant*100:.2f}% -- Disease Accuracy-- {acc_disease*100:.2f}%')
    f1score_disease.reset()

torch.save(model.state_dict(), 'multi_head_plant_disease.pth')