from DataProcessing import create_data, timing
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, models
from keras.utils import to_categorical
import matplotlib.pyplot as plt

image_size = 32

"Output of create_data() -> dict_keys([b'batch_label', b'labels', b'data', b'filenames'])"

test_data, categories = create_data("test")
validation_data, _ = create_data("validation")
#print(data[0], np.shape(data[0][0]))
num_categories = len(categories)

X_train, X_test = np.array(test_data[b'data']), np.array(validation_data[b'data'])
y_train_cat, y_test_cat = to_categorical(y_train, num_classes=num_categories), to_categorical(y_test, num_classes=num_categories)

#Short overview of test set
def show_overview():
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_train[i])
        plt.xlabel(categories[y_train[i]])
    print(f"Shape of X_train: {np.shape(X_train)} \n shape of X_test: {np.shape(X_test)}")
    print(f"Shape of y_train: {np.shape(y_train)} \n Shape of y_test: {np.shape(y_test)}")
    plt.show()

#show_overview()


#build model
model = models.Sequential(name="AnimalRecognition")
model.add(layers.Conv2D(128, (3, 3), padding="same", activation='relu', input_shape=(image_size, image_size, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(num_categories))
model.add(layers.Activation("softmax"))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

history = model.fit(X_train, y_train_cat, epochs=15, validation_data=(X_test, y_test_cat))

plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(X_test,  y_test_cat, verbose=2)
print(f"Test accuracy = {test_acc}")

plt.show()
