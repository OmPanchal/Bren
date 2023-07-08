from bren.nn.layers import Layer


class Activation(Layer):
	def __init__(self, func=None, name=None, **kwargs) -> None:
		self.func = func
		super().__init__(name, **kwargs)
		self.set_built(True)
		self.__class__.__name__ = func.__name__

	def call(self, inp, training=None, **kwargs):
		return self.func(inp)

	# def build(self, input_shape, input_dtype, **kwargs): ...

# def activation_from_func(func):
# 	class Activation(Layer):
# 		def __init__(self, name=func.__name__, **kwargs) -> None:
# 			super().__init__(name, **kwargs)
# 			print(func)
# 			self.set_built(True)

# 		def __call__(self, inp, training=None, **kwargs):
# 			return func(inp)

# 	return Activation