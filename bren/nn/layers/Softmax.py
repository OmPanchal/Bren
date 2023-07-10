from bren.nn.layers import Layer
import numpy as np
from bren.autodiff.operations.ops import custom_gradient
from bren.core.core import Variable


class Softmax(Layer):
    def __init__(self, name=None, **kwargs) -> None:
        super().__init__(name, **kwargs)

    def call(self, x):
        tmp = np.exp(x)
        return tmp / np.sum(tmp)