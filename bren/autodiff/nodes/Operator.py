import bren.autodiff.nodes as nodes


class Operator(nodes.Node):

	def __init__(self, inputs, value=None, name="Operator", **kwargs) -> None:
		self.inputs = inputs
		self.scalars = list(map(lambda x: x.value, self.inputs))
		self.grad = kwargs.get("grad")
		super().__init__(value, name)

	def __call__(self, dout): 
		# reset the gradient
		self.gradient = 0
		return self.grad(*self.scalars, dout=dout, value=self.value)

	def __repr__(self) -> str:
		return f"<{self.name.capitalize()} inputs={self.inputs}>"