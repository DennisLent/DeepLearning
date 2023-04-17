from DataProcessing import create_data
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, models
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import Tensorflow_models

image_size = 32 #default size

"Output of create_data() -> dict_keys([b'batch_label', b'labels', b'data', b'filenames'])"

train_data, train_labels, categories = create_data("Cifar10/test")
val_data, val_labels, _ = create_data("Cifar10/validation")
num_categories = len(categories)

X_train, X_test = train_data, val_data
y_train_cat, y_test_cat = to_categorical(train_labels, num_classes=num_categories), to_categorical(val_labels, num_classes=num_categories)

#model = Tensorflow_models.resnet(input_shape=(32,32,3), num_classes=num_categories)
model = Tensorflow_models.efficientNet(input_shape=(32,32,3), num_classes=num_categories)

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, mode="max")

history = model.fit(X_train, y_train_cat, epochs=30, validation_data=(X_test, y_test_cat), callbacks=early_stop)

plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(X_test,  y_test_cat, verbose=2)
print(f"Test accuracy = {test_acc}")

plt.show()

ans = input("save the model? [y/n]:")
if ans.lower() == "y":
    name_model = input("What should it be saved as?:")
    model.save(f"{name_model}.h5")
