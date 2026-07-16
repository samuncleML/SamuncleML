import torch
from PIL import Image
from module import PlantDiseaseModel
from torchvision import transforms

class PredictPlantDisease:
    def __init__(self, image_path: str, model_path : str,  plant_classes: list, disease_classes: list):
        self.model = self._load_model(model_path)
        self.plant_classes = plant_classes
        self.disease_classes = disease_classes
        self.image_path = image_path

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.426, 0.481, 0.344), (0.182, 0.184, 0.181))
        ])
    
    def _load_model(self, model_path):
        model = PlantDiseaseModel(6, 27)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    def _preprocess(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image).unsqueeze(0)
        return image
    
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
    def run_model(self):
        image_tensor = self._preprocess(image_path=self.image_path)

        with torch.no_grad():
            plant_outputs, disease_outputs = self.model(image_tensor)
            plant_probs = torch.softmax(plant_outputs, dim=1)
            disease_probs = torch.softmax(disease_outputs, dim=1)
        
        return self._format_results(plant_probs, disease_probs)
    
image_path = r"C:\Users\Administrator\Downloads\download (5).jpg"
model_path = r"C:\Users\Administrator\Documents\plant-disease-multhead\plant_disease_model (1).pth"
plant_classes = ['Cassava', 'Cocoa', 'Corn', 'Cowpea', 'Unknown']
disease_classes = ['Bacterial wilt', 'Cercospora_leaf_spot Gray_leaf_spot', 'Common_rust_', 
                   'Healthy Cassava', 'Healthy Cocoa', 'Healthy Corn', 'Healthy Cowpea', 
                    'Mosaic virus', 'Northern_Leaf_Blight','Septoria_leaf_spot','bacterial blight', 
                    'black_pod_rot', 'brown spot', 'Unknown disease', 'green mite', 'mosaic', 'pod_borer']

predict = PredictPlantDisease(image_path=image_path, model_path = model_path, plant_classes=plant_classes, disease_classes=disease_classes)
print(predict.run_model())