import numpy as np
from bren.nn.layers.Layer import Layer


def tanh(x): return np.tanh(x)

class Tanh(Layer):
	def __init__(self, name=None, **kwargs) -> None:
		super().__init__(name, **kwargs)

	def call(self, x): return tanh(x)