from DataProcessing import create_data, timing
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, models
from keras.utils import to_categorical
import matplotlib.pyplot as plt

image_size = 64

data, categories = create_data(image_size)
#print(data[0], np.shape(data[0][0]))
num_categories = len(categories)

X = []
y = []

for features, label in data:
    #print(f"picture array length: {len(features)}")
    #print(f"classification: {label}")
    X.append(features)
    y.append(label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=23)
X_train, X_test = np.array(X_train), np.array(X_test)
y_train, y_test = to_categorical(y_train, num_classes=num_categories), to_categorical(y_test, num_classes=num_categories)


#build model
model = models.Sequential(name="AnimalRecognition")
model.add(layers.Conv2D(32, (4,4), padding="same", input_shape=(image_size, image_size, 3)))
model.add(layers.Activation("relu"))
model.add(layers.Conv2D(128, (4,4)))
model.add(layers.Activation("relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(256))
model.add(layers.Activation("relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_categories))
model.add(layers.Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=10, batch_size=32)

plt.plot(history.history["accuracy"], label="accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
