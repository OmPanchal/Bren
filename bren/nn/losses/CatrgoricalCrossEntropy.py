from bren.nn.losses import Loss
import numpy as np


def categorical_cross_entropy(y_pred, y_true, epsilon=1e-8): 
	return -(np.sum(y_true * np.log(y_pred + epsilon)))

class CategoricalCrossEntropy(Loss):
	"""
	`CategoricalCrossEntropy` computes the loss for multiclass classification, with `loss=-(sum(y_true * log(y_pred + epsilon)))` 
	"""

	def __init__(self) -> None:
		super().__init__()

	def __call__(self, y_pred, y_true, *args, **kwargs):
		return categorical_cross_entropy(y_pred, y_true)
