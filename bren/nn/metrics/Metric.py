from bren import Variable
import numpy as np


class Metric(object):
	def __init__(self) -> None: 
		self.__reset = np.vectorize(self.__reset, cache=False)
		self.variables = []

	def __reset(self, var): var.assign(0) 

	def update(self, y_pred, y_true, weights=None): ...

	def reset(self): self.__reset(self.variables)

	def add_weight(self, val, **kwargs):
		var = Variable(val, **kwargs)
		# if kwargs.get("trainable") is not False: self.trainable.append(var)
		self.variables.append(var)
		return var
	
	def result(self):...

	def __call__(self, y_pred, y_true, **kwargs): 
		self.update(y_pred, y_true)
		return self.result(**kwargs)

