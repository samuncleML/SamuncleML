import torch
import numpy as np
import torch.nn as nn
from torchvision import models

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_plant_classes: int, num_disease_classes :int):
        super(PlantDiseaseModel, self).__init__()
        base_model = models.mobilenet_v3_small(weights='DEFAULT')
        self.shared = nn.Sequential(
            base_model.features,
            base_model.avgpool
        )

        in_features = 576
        self.disease_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 1024),

            nn.Hardswish(),
            nn.Dropout(0.3, inplace=True),
            nn.Linear(1024, num_disease_classes)
        )

        self.plant_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 1024),

            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_plant_classes)
        )
    
    def forward(self, x):

        features = self.shared(x)
        plant_outputs = self.plant_head(features)
        disease_outputs = self.disease_head(features)

        return plant_outputs, disease_outputs