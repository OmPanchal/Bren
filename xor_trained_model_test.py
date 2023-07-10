import bren as br
from AddLayer import AddLayer
import numpy as np


def belu(x): return x

def random(shape, **kwargs): 
	print(shape)
	return br.Variable(np.random.random(size=shape))

layer_pool = {"AddLayer": AddLayer,
              "belu": belu,
	      "random": random}


X = br.Variable([[1, 1], [1, 0], [0, 1], [0, 0]], dtype="float64")
Y = br.Variable([0, 1, 1, 0], dtype="float64")

model = br.nn.models.load_model("test_model", custom_objects=layer_pool)

print(model.predict(X))