from bren.nn.initialisers import Initialiser
import numpy as np


class GlorotNormal(Initialiser):
	def __init__(self, shape, dtype="float32", **kwargs) -> None:
		super().__init__(shape, dtype, **kwargs)

		self.mean = kwargs.get("mean") or 0
		self.fan_in, self.fan_out = kwargs.get("inout") or shape
		fan_avg = (self.fan_in + self.fan_out) / 2
		self.variance = (1 / fan_avg)

	def call(self, *args, **kwargs):
		return np.random.normal(self.mean, self.variance, size=self.shape)
