from tensorflow import keras
from keras.models import Sequential
from keras import Input
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report as cr

#Set up the data

file = "cardio.csv"
df = pd.read_csv(file, delimiter=";")

x = df[["age", "gender", "height", "weight", "ap_hi", "ap_lo", "smoke", "alco", "active"]]
y = df["cardio"]


x_train, x_test, y_train, y_test = tts(x, y, test_size=0.33, random_state=23)

#Set up Feed Forward Network 
model = Sequential(name = "CardiovasularDetection")
model.add(Input(shape=(9,), name="input"))
model.add(Dense(16, activation="sigmoid", name="hidden1"))
model.add(Dense(8, activation="sigmoid", name="hidden2"))
model.add(Dense(8, activation="sigmoid", name="hidden3"))
model.add(Dense(4, activation="sigmoid", name="hidden4"))
model.add(Dense(1, activation="sigmoid", name="output"))

#train model
model.compile(loss="binary_crossentropy", metrics=['Accuracy', 'Precision', 'Recall'])
model.fit(x_train, y_train, epochs=10)

prediction_train = (model.predict(x_train) > 0.5).astype(int)
prediction_test = (model.predict(x_test) > 0.5).astype(int)

#summary
model.summary()

for layer in model.layers:
    print(f"Layer: {layer}")
    print(f"Kernels (weights): {layer.get_weights()[0]}")
    print(f"Biases: {layer.get_weights()[1]}")

print("TRAINGING EVALUATION")
print(cr(y_train, prediction_train))

print("TESTING EVALUATION")
print(cr(y_test, prediction_test))
