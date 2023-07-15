from bren.nn.losses import Loss
import numpy as np


def mse(y_pred, y_true): return np.mean((y_pred - y_true) ** 2)

class MeanSquaredError(Loss):
	"""
	`MeanSquaredError` computes the loss as `loss=mean((y_pred - y_true) ** 2)`
	"""
	
	def __init__(self) -> None:
		super().__init__()

	def __call__(self, y_pred, y_true, *args, **kwargs):
		return mse(y_pred, y_true)

MSE = MeanSquaredError
mean_squared_error = mse