import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, models
from keras.utils import to_categorical
import matplotlib.pyplot as plt

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