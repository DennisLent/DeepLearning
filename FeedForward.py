from tensorflow import keras
from keras.models import Sequential
from keras import Input
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts

#Set up the data

df = pd.read_csv("cardio.csv", delimiter=";")
print(df)

x = df[["age", "gender", "height", "weight", "ap_hi", "ap_lo", "smoke", "alco", "active"]]
y = df["cardio"]


x_train, x_test, y_train, y_test = tts(x, y, test_size=0.33, random_state=23)
print(x_train, y_train)

#Set up Feed Forward Network 



