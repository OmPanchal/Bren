from bren.nn.layers import Layer


class Activation(Layer):
	def __init__(self, func=None, name=None, **kwargs) -> None:
		self.func = func
		super().__init__(name, **kwargs)
		self.set_built(True)
		self.__class__.__name__ = func.__name__

	def call(self, inp, training=None, **kwargs):
		return self.func(inp)
