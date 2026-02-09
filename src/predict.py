import torch
from PIL import Image
from Module import PlantDiseaseModel
from torchvision import transforms
import cv2

class PredictPlantDisease:
    def __init__(self, image_path: str, model_path : str,  plant_classes: list, disease_classes: list):
        self.model = self._load_model(model_path)
        self.plant_classes = plant_classes
        self.disease_classes = disease_classes
        self.image_path = image_path

        self.transforms = transforms.Compose([
            transforms.RandomAdjustSharpness(1.5),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop((320, 320), scale=(0.2, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.432, 0.463, 0.381), (0.175, 0.166, 0.187))
        ])
    
    def _load_model(self, model_path):
        model = PlantDiseaseModel(5, 17)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    def preprocess(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image).unsqueeze(0)
        return image
    
    def run_model(self):
        image_tensor = self.preprocess(image_path=self.image_path)

        with torch.no_grad():
            plant_outputs, disease_outputs = self.model(image_tensor)
            plant_probs = torch.softmax(plant_outputs, dim=1)
            disease_probs = torch.softmax(disease_outputs, dim=1)
        
        return self._format_results(plant_probs, disease_probs)
    
    def _format_results(self, p_probs, d_probs):
        plant_idx = torch.argmax(p_probs)
        plant_confidence = p_probs[0][plant_idx].item()
        plant_name = self.plant_classes[plant_idx]

        if plant_name == 'Unknown' and plant_confidence >= 0.7:
            return {'status':'error', 'message':'Unknown object detected'}
        
        disease_idx = torch.argmax(d_probs)
        disease_confidence = d_probs[0][disease_idx].item()
        disease_name = self.disease_classes[disease_idx]

        return {
            'status':'success', 'plant':plant_name, 'disease':disease_name,
            'confidence':{'plant':plant_confidence, 'disease':disease_confidence}
        }

image_path = r"C:\Users\Administrator\Downloads\download (4).jpg"
model_path = r'C:\Users\Administrator\Documents\Project\multi_head_plant_disease.pth'
plant_classes = ['Cassava', 'Cocoa', 'Corn', 'Cowpea', 'Unknown']
disease_classes = ['Bacterial wilt', 'Cercospora_leaf_spot Gray_leaf_spot', 'Common_rust_', 
                   'Healthy Cassava', 'Healthy Cocoa', 'Healthy Corn', 'Healthy Cowpea', 
                    'Mosaic virus', 'Northern_Leaf_Blight','Septoria_leaf_spot','bacterial blight', 
                    'black_pod_rot', 'brown spot', 'Unknown disease', 'green mite', 'mosaic', 'pod_borer']

predict = PredictPlantDisease(image_path=image_path, model_path = model_path, plant_classes=plant_classes, disease_classes=disease_classes)
print(predict.run_model())