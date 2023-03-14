from DataProcessing import create_data
import random
import numpy as np

data = create_data()
print(data[0])
random.shuffle(data)

X = []
y = []

for features, label in data:
    #print(f"picture array length: {len(features)}")
    #print(f"classification: {label}")
    X.append(features)
    y.append(label)