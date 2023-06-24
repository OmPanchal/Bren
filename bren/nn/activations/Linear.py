from bren.nn.layers.Layer import Layer


class Linear(Layer):
	def __init__(self, name=None, **kwargs) -> None:
		super().__init__(name, **kwargs)

	def call(self, x): return x
