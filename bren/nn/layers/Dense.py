from bren.nn.layers import Layer
from bren.nn.initialisers import get_initialiser, Initialiser, initialiser_from_func
from bren.nn.activations import get_activation


def set_activation(activation):
	out = None

	if type(activation) == str or activation is None: 
		out = get_activation(activation)()
	elif issubclass(type(activation), Layer) or type(activation).__name__ == function.__name__:
		out = activation
	else: raise AttributeError(f"{activation} is of invalid type for activation")

	return out

def set_initialiser(initialiser):
	out = None

	if type(initialiser) == str:
		out = get_initialiser(initialiser)
	elif issubclass(type(initialiser), Initialiser):
		out = initialiser 
	elif type(initialiser).__name__ == function.__name__:
		out = initialiser_from_func(initialiser)
	else: raise AttributeError(f"{initialiser} is of invalid type for initialisers")

	return out
	

class Dense(Layer):
	def __init__(self, units, 
						activation=None, 
						weights_initialiser="GlorotUniform", 
						bias_initialiser="GlorotUniform", 
						use_bias=True, 
						name=None, **kwargs) -> None:
		super().__init__(name, **kwargs)

		self._units = units
		self.activation = set_activation(activation)

		self.weights_initialiser = set_initialiser(weights_initialiser)
		self.bias_initialiser = set_initialiser(bias_initialiser)

		self.use_bias = use_bias

	@property
	def units(self): return self._units
	@units.setter
	def units(self, value): ...

	def build(self, input_shape, input_dtype, **kwargs):
	
		self.weights = self.add_weight(
			self.weights_initialiser(shape=(self.units, input_shape[0]))(), dtype="float64") 

		self.bias = self.add_weight(
			self.bias_initialiser(shape=(self.units, 1))() if self.use_bias else 0, dtype="float64")

		print(self.weights.dtype, self.bias.dtype)
		# print(self.weights.shape)
		# print(self.bias.shape)
		return super().build(input_shape, input_dtype, **kwargs)

	def call(self, x):
		output = (self.weights @ x) + self.bias
		
		return self.activation(output)

# aliases
FullyConnected = Dense
FC = Dense