import bren as br


class AddLayer(br.nn.layers.Layer):
	def __init__(self, name=None, **kwargs) -> None:
		super().__init__(name, **kwargs)
		self.weight = self.add_weight(69)
		
	def call(self, x): 
		return 69 * (x + x)