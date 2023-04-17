from DataProcessing import create_data, timing
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, models
from keras.utils import to_categorical
import matplotlib.pyplot as plt

image_size = 32 #default size

"Output of create_data() -> dict_keys([b'batch_label', b'labels', b'data', b'filenames'])"

train_data, train_labels, categories = create_data("test")
val_data, val_labels, _ = create_data("validation")
num_categories = len(categories)

X_train, X_test = train_data, val_data
y_train_cat, y_test_cat = to_categorical(train_labels, num_classes=num_categories), to_categorical(val_labels, num_classes=num_categories)


#build model
model = models.Sequential(name="AnimalRecognition")
model.add(layers.Conv2D(32, (3, 3), padding="valid", activation='relu', input_shape=(image_size, image_size, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), padding="valid", activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128, (3, 3), padding="valid", activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(num_categories, activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

history = model.fit(X_train, y_train_cat, batch_size=128, epochs=30, validation_data=(X_test, y_test_cat))

plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(X_test,  y_test_cat, verbose=2)
print(f"Test accuracy = {test_acc}")

plt.show()
