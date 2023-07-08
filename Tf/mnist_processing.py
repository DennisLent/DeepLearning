import os
from functools import wraps
from time import time
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

# # Create the image folder if it doesn't exist
# if not os.path.exists(image_folder):
#     os.makedirs(image_folder)

# with open(csv_file, 'r') as file:
#     csv_reader = csv.reader(file)
#     next(csv_reader)  # Skip the header row

#     for row in csv_reader:
#         label = int(row[0])
#         pixels = np.array(row[1:], dtype=np.uint8).reshape(28, 28)
#         image = Image.fromarray(pixels)

#         # Save the image with the associated label
#         image_path = os.path.join(image_folder, f'image_{label}.png')
#         image.save(image_path)

