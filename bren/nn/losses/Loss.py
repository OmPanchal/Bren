from bren.nn.metrics import Metric
import numpy as np

class Loss(Metric):
	"""
	The base `Loss` class
	"""

	def __init__(self, **kwargs) -> None: 
		self.func = kwargs.get("func", None)
	def __call__(self, y_pred, y_true, *args, **kwargs): 
		out = self.func(y_pred, y_true, *args, **kwargs)
		return np.sum(out)
		