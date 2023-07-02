from bren.nn.layers import Layer
import numpy as np


class Flatten(Layer):
    def __init__(self, name=None, **kwargs) -> None:
        super().__init__(name, **kwargs)

    def call(self, x):
        return x.flatten()[..., np.newaxis]
    

