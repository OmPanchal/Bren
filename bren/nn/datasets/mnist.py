import numpy as np
import requests, gzip, os, pathlib
from bren import Variable


#fetch data
path = os.path.join(pathlib.Path().resolve(), "bren\\nn\\datasets\\data")

def fetch(url):
    if os.path.isfile(path):
        with open(path, "rb") as f:
            data = f.read()
    else:
        with open(path, "wb") as f:
            data = requests.get(url).content
            f.write(data)
    os.remove(path)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()


def load_mnist(dtype="float64"):
	X = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
	Y = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
	X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
	Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]
	return (Variable(X, dtype=dtype), Variable(Y, dtype=dtype), 
         Variable(X_test, dtype=dtype), Variable(Y_test, dtype=dtype))
