from bren.nn.metrics import Metric

class Loss(Metric):
	"""
	The base `Loss` class
	"""

	def __init__(self) -> None: ...
	def __call__(self, *args, **kwargs): ...