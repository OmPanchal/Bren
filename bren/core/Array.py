import numpy as np
from bren.autodiff.nodes import Graph
from bren.autodiff.operations.ops import OPS
import typing


def make_ops_source(name, **kwargs):
	try: 
		if Graph._g: return OPS.get(name, OPS["nongrad"])(**kwargs)
		else: return None
	except KeyError:
		raise RuntimeError(f"no gradient found for operation {name}")


class Array(np.lib.mixins.NDArrayOperatorsMixin, typing.Sequence):
	def __init__(self, value, dtype="float32", name=None, **kwargs) -> None:
		self.dtype = dtype
		if issubclass(type(value), Array):
			self._i = np.array(value._i, dtype=self.dtype)
		else: self._i = np.array(value, dtype=self.dtype)
		self.shape = self._i.shape
		self.size = self._i.size
		self.name = name
		self.frozen = False
		super().__init__()

	@property
	def length(self): return self.shape[0]
	@length.setter
	def length(self): ...

	def freeze(self): self.frozen = True
	def unfreeze(self): self.frozen = False

	def __len__(self): 
		if not self.frozen: return self.shape[0]

	def __repr__(self) -> str:
		return f"<{self.__class__.__name__} value={self._i} dtype={self.dtype}>"
	
	def __getitem__(self, key):
		return self.__class__(self._i[key], dtype=self.dtype)
	def __setitem__(self, key, value):
		self._i[key] = value

	def __iter__(self):
		self._idx = 0
		return self
	
	def __next__(self):
		if self._idx < len(self._i):
			x = self._i[self._idx]
			self._idx += 1
			return x
		raise StopIteration

	def __array_function__(self, func, types, args, kwargs):
		scalars, sources = self.__set_source_scalar(args)

		val = func(*scalars, **kwargs)
		source = make_ops_source(func.__name__, inputs=sources, value=val)

		return self.__class__(val, dtype=self.dtype, source=source)

	def __array_ufunc__(self, ufunc, method, *inps, **kwargs):
		if method == "__call__":
			scalars, sources = self.__set_source_scalar(inps)

		val = ufunc(*scalars, **kwargs)
		source = make_ops_source(ufunc.__name__, inputs=sources, value=val)
		return self.__class__(val, dtype=self.dtype, source=source, )	

	def __set_source_scalar(self, inps):
		scalars = []
		sources = []

		for inp in inps:
			val = inp
			
			if not issubclass(type(inp), self.__class__):
				try: val = self.__class__(inp)
				except:
					raise ValueError(f"Cannot convert {inp} into type {self.__class__.__name__}")

			scalars.append(val._i)
			sources.append(val._source)

		return scalars, sources

	def append(self, values, **kwargs):
		self._i = np.append(self._i, values, *kwargs)

	def assign(self, value):
		if issubclass(type(value), Array): value = value._i
		self._i = value
		self._source.value = self._i

		return self._i

	def assign_add(self, value): return self.assign(self._i + value)
	def assign_sub(self, value): return self.assign(self._i - value)

	def clone(self): return self.__class__(self._i, self.dtype, self.name)

	def numpy(self, dtype=None): return self._i.astype(dtype or self.dtype)

	def flatten(self): return self._i.flatten()

	def astype(self, dtype): 
		self.dtype = dtype
		return self

	@property
	def T(self): return np.transpose(self)
	@T.setter
	def T(self): ...

