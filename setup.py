import os
import urllib.request
import tarfile
import gdown
import zipfile

url_cifar = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
filename_cifar = "cifar-10-python.tar.gz"
url_mnist = "https://drive.google.com/file/d/1jiFRYsI6UczzVn5uUj-zMA_PNlBhlkvX/view?usp=sharing"
filename_mnist = "mnist"

if not os.path.exists("cifar-10-batches-py"):
    print(f"downloading {filename_cifar}...")
    urllib.request.urlretrieve(url_cifar, filename_cifar)

with tarfile.open(filename_cifar, "r:gz") as tar:
    print(f"extracting {filename_cifar}...")
    tar.extractall()

os.remove(filename_cifar)

if not os.path.exists("mnist"):
    os.makedirs("mnist")
    print(f"downloading {filename_mnist}...")
    gdown.download(url_mnist, output="mnist.zip", quiet=False)

    output = zipfile.ZipFile("mnist.zip")
    output.extractall("mnist")
