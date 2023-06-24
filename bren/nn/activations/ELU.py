import numpy as np
from bren.nn.layers.Layer import Layer
from bren.autodiff.operations.ops import custom_gradient


def elu_grad(a, alpha, dout, value):
	return [np.multiply(dout, np.where(a > 0, 1, alpha * np.exp(a)))]
@custom_gradient(elu_grad)
def elu(a, alpha): 
	return np.where(a > 0, a, alpha * (np.exp(a) - 1))


class ELU(Layer):
	def __init__(self, alpha=1, name=None, **kwargs) -> None:
		self.alpha = alpha
		super().__init__(name, **kwargs)

	def call(self, x): return elu(x, self.alpha)