import numpy as np


class Optimiser(object):
	def __init__(self) -> None: 
		self.update = np.vectorize(self.update, cache=True, otypes=[list])

	def apply_gradients(self, vars, grads, **kwargs): ...

	def update(*args, **kwargs): ...