import os
import urllib.request
import tarfile

url_cifar = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
filename_cifar = "cifar-10-python.tar.gz"

if not os.path.exists("cifar-10-batches-py"):
    urllib.request.urlretrieve(url_cifar, filename_cifar)

with tarfile.open(filename_cifar, "r:gz") as tar:
    tar.extractall()

os.remove(filename_cifar)