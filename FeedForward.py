from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts

#setting up the data

df = pd.read_csv("seattle-weather.csv")
df["weather"].replace(['drizzle', 'rain', 'sun', 'snow', 'fog'], [0,1,2,3,4], inplace=True)

x = df[["precipitation", "temp_max", "temp_min", "wind"]]
y = df["weather"]

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.33, random_state=23)