import os
from os import listdir
from os import path
import cv2
from functools import wraps
from time import time
import pickle

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
        print(f"unpickling {filename}")
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            dict = unpickle(f)
            print(f"unpickled dictionary: {dict}")
            all_data.update(dict)
    
    return all_data


if __name__ == "__main__":
    test_data = create_data("test")
    print(f"Length of dict = {len(test_data)}")