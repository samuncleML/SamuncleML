from Module import PlantDiseaseModel
import onnxscript
import torch

model = PlantDiseaseModel(5, 29)
checkpoint = torch.load(r'C:\Users\Administrator\Documents\plant-disease-multihead\models\dual_head_mobilenetv3_small.pt', map_location=torch.device('cpu'))
weights    = checkpoint['model_state_dict']
model.load_state_dict(weights)

model.eval()
image = torch.rand((1, 3, 224, 224))
onnx_program = torch.onnx.export(model, image, dynamo=True)
onnx_program.save(r'C:\Users\Administrator\Documents\plant-disease-multihead\models\dual_head_mobilenetv3_small.onnx')

print(model)