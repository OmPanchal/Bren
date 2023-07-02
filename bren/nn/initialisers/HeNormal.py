import numpy as np
from bren.nn.initialisers.Initialiser import Initialiser


class HeNormal(Initialiser):
	def __init__(self, shape, dtype="float32", **kwargs) -> None:
		super().__init__(shape, dtype, **kwargs)

		self.mean = kwargs.get("mean") or 0
		self.fan_in, _ = kwargs.get("inout") or shape
		self.variance = 2 / self.fan_in

	def call(self, *args, **kwargs):
		return np.random.normal(self.mean, self.variance, size=self.shape)