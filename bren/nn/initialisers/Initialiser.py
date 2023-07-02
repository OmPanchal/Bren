class Initialiser(object):
	def __init__(self, shape, dtype="float64", **kwargs) -> None:
		self.shape = shape
		self.dtype = dtype

	def __call__(self, *args, **kwargs): 
		return self.call(**self.__dict__).astype(self.dtype)

	def call(self, shape, *args, **kwargs):...


def initialiser_from_func(func):
	class Initialiser(object):
		def __init__(self, shape, dtype="float64", **kwargs) -> None:
			self.shape = shape
			self.dtype = dtype

		def __call__(self, *args, **kwargs): 
			return func(**self.__dict__).astype(self.dtype)

	return Initialiser