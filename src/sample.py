import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('src/data/raw/crop_disease_labels.csv')

train, sample = train_test_split(data, stratify=data['disease'], test_size=0.2)
print(sample['disease'].value_counts())

test, val = train_test_split(sample, stratify=sample['disease'], test_size=0.5)

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
val.to_csv('val.csv', index=False)
