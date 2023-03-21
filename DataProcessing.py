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

root = "C:/Users/Dennis Lentschig/Desktop/Python/DeepLearning/archive"



@timing
def create_data(image_size = 75):
    training_data = []
    for category in categories:
        path = os.path.join(root, category)
        class_indentifier = categories.index(category)
        for image in os.listdir(path):
            try:
                if image.endswith('.png'):
                    os.remove(image) 
                image_array = cv2.imread(os.path.join(path, image))
                new_array = cv2.resize(image_array, (image_size, image_size))
                #print(new_array)
                #cv2.imshow(f"{class_indentifier}", new_array)
                #cv2.waitKey(0)
                training_data.append([new_array, class_indentifier])
            except Exception as e:
                pass
    return training_data, categories

if __name__ == "__main__":
    create_data(image_size = 150)