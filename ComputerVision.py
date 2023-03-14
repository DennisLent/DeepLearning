from DataProcessing import create_data
import random
import numpy as np
from sklearn.model_selection import train_test_split

data = create_data()
print(data[0], np.shape(data[0][0]))

X = []
y = []

for features, label in data:
    #print(f"picture array length: {len(features)}")
    #print(f"classification: {label}")
    X.append(features)
    y.append(label)