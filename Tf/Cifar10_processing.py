import os
from os import listdir
from os import path
from functools import wraps
from time import time
import pickle
import matplotlib.pyplot as plt
import numpy as np

def timing(f):
    """
    wrapper function to time (just for my sake)
    """
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"sampling took {te-ts} sec")
        return result
    return wrap

@timing
def create_data(directory, type):
    """
    Unpickle the pickled pictures with their lables
    """
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    def make_image(image_array):
        r = image_array[:1024].reshape(32,32) / 255.0
        g = image_array[1024:2048].reshape(32,32) / 255.0
        b = image_array[2048:].reshape(32,32) / 255.0
        rgb = np.stack([r,g,b], axis=-1)
        return rgb
    
    all_data = {}
    label_dict = {0: "airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if (type in filename) and (os.path.isfile(f)):
            dict = unpickle(f)
            all_data.update(dict)

    data, labels = [], []
    
    for i in range(len(all_data[b'data'])):
        image_array = np.array(all_data[b'data'][i])
        image_array = make_image(image_array)
        image_label = all_data[b'labels'][i]
        
        data.append(image_array)
        labels.append(image_label)

    
    return np.asarray(data), np.asarray(labels), label_dict


if __name__ == "__main__":
    test_data, test_labels, label_dict = create_data("cifar-10-batches-py", "test")

    plt.figure(figsize=(10,10))
    for i in range(36):
        plt.subplot(6,6,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        image_array = test_data[i]
        plt.imshow(image_array)
        plt.xlabel(label_dict[test_labels[i]])
    plt.show()



