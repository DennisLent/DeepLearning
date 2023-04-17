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
def conv_block(x, filters, kernel_size, strides):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def identity_block(x, filters, kernel_size):
    x_shortcut = x
    x = conv_block(x, filters, kernel_size, strides=(1, 1))
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x_shortcut, x])
    x = tf.keras.layers.ReLU()(x)
    return x

def resnet(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = conv_block(inputs, filters=64, kernel_size=(7, 7), strides=(2, 2))
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = identity_block(x, filters=64, kernel_size=(3, 3))
    x = identity_block(x, filters=64, kernel_size=(3, 3))
    x = conv_block(x, filters=128, kernel_size=(1, 1), strides=(2, 2))
    x = identity_block(x, filters=128, kernel_size=(3, 3))
    x = identity_block(x, filters=128, kernel_size=(3, 3))
    x = conv_block(x, filters=256, kernel_size=(1, 1), strides=(2, 2))
    x = identity_block(x, filters=256, kernel_size=(3, 3))
    x = identity_block(x, filters=256, kernel_size=(3, 3))
    x = conv_block(x, filters=512, kernel_size=(1, 1), strides=(2, 2))
    x = identity_block(x, filters=512, kernel_size=(3, 3))
    x = identity_block(x, filters=512, kernel_size=(3, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = resnet(input_shape=(32,32,3), num_classes=num_categories)

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, mode="max")

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
