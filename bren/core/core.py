import numpy as np
from bren.autodiff.nodes import Var, Const
from bren.core.Array import Array


class Variable(Array):
	def __init__(self, value, dtype="float32", name=None, **kwargs) -> None:
		super().__init__(value, dtype, name, **kwargs)
		self._source = kwargs.get("source") or Var(self._i, name=self.name)


class Constant(Array):
	def __init__(self, value, dtype="float32", name=None, **kwargs) -> None:
		super().__init__(value, dtype, name, **kwargs)
		self._source = kwargs.get("source") or Const(self._i, name=self.name)
