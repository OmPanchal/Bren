from bren import Variable
from bren.nn.utils import rename_key


class Layer(object):
	"""
	The base `Layer` class.

	Parameters
	----------
	name (`str`): The name of the layer.
	"""

	count = 0

	def __init__(self, name=None, **kwargs) -> None:
		self.custom_objs = kwargs.get("custom_obs")
		self.trainable = kwargs.get("params", {})
		self.name = name or self.set_name(Layer)
		self.built = False
		Layer.count += 1

	def __call__(self, inp, training=None, **kwargs): 
		# print(self.name, self.__class__, self.built)
		if not self.built:
			# print("This was called")
			self.build(input_shape=inp.shape, input_dtype=inp.dtype)
		return self.call(inp)
		 
	def __delattr__(self, __name) -> None: ...
	
	def build(self, input_shape, input_dtype, **kwargs):
		for key, value in zip(self.__dict__.keys(), self.__dict__.values()):
			if type(value) is Variable: self.trainable[key] = value
		
		self.built = True

	def set_built(self, bool): self.built = bool

	def call(self, x): ...

	def add_weight(self, val, **kwargs):
		var = Variable(val, **kwargs)
		# self.trainable.append(var)
		# if kwargs.get("trainable") is not False: self.trainable.append(var)
		return var

	def set_weights(self, params):
		self.trainable = params

	def get_weights(self):
		return self.trainable

	def set_name(self, cls): return f"{cls.__name__}/{cls.count}"

	def set_config(self, dictionary): self.__dict__ = dictionary 

	def config(self) -> dict:
		if "layer" in self.__dict__.keys():
			raise AttributeError(f"The attribute 'layer' is reserved for the model's internals, please change the attribute name")

		self.__dict__ = {"layer": self.__class__.__name__, **self.__dict__}
		return {"layer": self.__class__.__name__, **self.__dict__}