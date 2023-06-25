from bren import Variable


class Layer(object):
	count = 0

	def __init__(self, name=None, **kwargs) -> None:
		self.trainable = kwargs.get("params") or []
		self.name = name or self.set_name(Layer)
		self.__built = False
		Layer.count += 1

	def __call__(self, inp, training=None, **kwargs): 
		if not self.__built:
			self.build(input_shape=inp.shape, input_dtype=inp.dtype)
		return self.call(inp)
		 
	def __delattr__(self, __name) -> None: ...
	
	def build(self, input_shape, input_dtype, **kwargs):
		self.__built = True

	def call(self, x): ...

	def add_weight(self, val, **kwargs):
		var = Variable(val, **kwargs)
		# if kwargs.get("trainable") is not False: self.trainable.append(var)
		self.trainable.append(var)
		return var

	def set_weights(self, params):
		self.trainable = params

	def get_weights(self):
		return self.trainable

	def set_name(self, cls): return f"{cls.__name__}/{cls.count}"