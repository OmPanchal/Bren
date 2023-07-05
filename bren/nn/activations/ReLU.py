import numpy as np
from bren.nn.layers.Layer import Layer
from bren.autodiff.operations.ops import custom_gradient


def relu_grad(a, leak, dout, value): 
	return [np.multiply(dout, a > 0)]
@custom_gradient(relu_grad)
def relu(a, leak): return np.maximum(a * leak, a)


class ReLU(Layer):
	def __init__(self, leak=0, name=None, **kwargs) -> None:
		self.leak = leak
		super().__init__(name, **kwargs)

	def call(self, x): return relu(x, self.leak)