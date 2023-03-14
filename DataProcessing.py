import os
from os import listdir
from os import path
import cv2
from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"function took {te-ts} sec")
        return result
    return wrap

root = "C:/Users/Dennis Lentschig/Desktop/Python/DeepLearning/archive"

image_size = 75

categories = ["Butterfly", "Cat", "Chicken", "Cow", "Dog", "Elephant", "Horse", "Sheep", "Spider", "Squirrel"]


@timing
def create_data():
    training_data = []
    for category in categories:
        path = os.path.join(root, category)
        class_indentifier = categories.index(category)
        for image in os.listdir(path):
            image_array = cv2.imread(os.path.join(path, image))
            new_array = cv2.resize(image_array, (image_size, image_size))
            new_array_bw = cv2.cvtColor(new_array, cv2.COLOR_BGR2GRAY)
            #cv2.imshow(f"{class_indentifier}", new_array_bw)
            #cv2.waitKey(0)
            training_data.append([new_array_bw, class_indentifier])
    return training_data
