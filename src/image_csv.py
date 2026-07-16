import os
import pandas as pd

BASE_PATH = os.getcwd()

path = os.path.join(BASE_PATH, 'plant')
plants = os.listdir(path)

plant_ = []
disease_ = []
image_ = []

for plant in plants:
    plant_path = os.path.join(path, plant)

    for disease in os.listdir(plant_path):
        for image in os.listdir(os.path.join(plant_path, disease)):
            image = os.path.join(plant_path, disease, image)

            plant_.append(plant)
            disease_.append(disease.split('__')[-1])
            image_.append(image)
print(len(plant_), len(disease_), len(image_))

image_dict = {'plant':plant_, 'disease':disease_, 'image':image_}
Image_data = pd.DataFrame(image_dict)
image_csv = Image_data.to_csv(os.path.join(BASE_PATH, 'src/data/raw/crop_disease_labels.csv'), index=False)

print("CSV created successfully with", len(Image_data), "entries.")
