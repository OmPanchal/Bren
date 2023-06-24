from bren.nn.metrics import Metric
import numpy as np


class Accuracy(Metric):
	def __init__(self) -> None:
		super().__init__()
		self.total = self.add_weight(0)
		self.count = self.add_weight(0)
		self.leniency = 2

	def update(self, y_pred, y_true, weights=None, **kwargs):
		acc = np.sum(y_true == np.round(y_pred, decimals=self.leniency)) / len(y_true)
		# print(y_pred, y_true)
		self.total.assign_add(acc)
		self.count.assign_add(1)

	def result(self):
		return self.total / self.count
	