import os
import urllib.request
import tarfile

url_cifar = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
filename = "cifar-10-python.tar.gz"

if not os.path.exists("Cifar10"):
    urllib.request.urlretrieve(url_cifar, filename)

with tarfile.open(filename, "r:gz") as tar:
    tar.extractall()