import os
from os import listdir
from os import path
import cv2
from functools import wraps
from time import time
import pickle
import matplotlib.pyplot as plt
import numpy as np

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"sampling took {te-ts} sec")
        return result
    return wrap

@timing
def create_data(directory):

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    all_data = {}

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            dict = unpickle(f)
            all_data.update(dict)
    
    return all_data


if __name__ == "__main__":
    test_data = create_data("test")
    print(f"keys of the dictionary = {test_data.keys()}")
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        image_array = np.array(test_data[b'data'][i])
        image_array = image_array.reshape((32,32,3))
        print(f"image array = {image_array}")
        plt.imshow(image_array.astype(np.uint8))
    plt.show()



