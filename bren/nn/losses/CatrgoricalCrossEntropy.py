from bren.nn.losses import Loss
import numpy as np


def categorical_cross_entropy(y_pred, y_true, epsilon=1e-8): 
	return -(np.sum(y_true * np.log(y_pred + epsilon)))

class CategoricalCrossEntropy(Loss):
	def __init__(self, epsilon=1e-8) -> None:
		super().__init__()
		self.epsilon = epsilon 

	def __call__(self, y_pred, y_true, *args, **kwargs):
		return categorical_cross_entropy(y_pred, y_true, epsilon=self.epsilon)
