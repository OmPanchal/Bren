from bren.nn.metrics import Metric
import numpy as np


class MeanSquaredError(Metric):
	def __init__(self) -> None:
		super().__init__()
		self.total = self.add_weight(0)
		self.count = self.add_weight(0)
		
	def update(self, y_pred, y_true, weights=None, **kwargs): 
		self.total.assign_add(kwargs.get("loss", np.sum((y_true - y_pred) ** 2))) 
		# ^ either give a value for the loss ei from the loss function or class, or calculate it if none given ... 
		self.count.assign_add(1)

	def result(self): return self.total / self.count
