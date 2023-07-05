from bren.nn.layers import Layer
import numpy as np
from bren.autodiff.operations.ops import custom_gradient
from bren.core.core import Variable


# def softmax_grad(x, dout, value):
#     n = value.size
#     tmp = np.tile(value, n)
#     return [np.dot(tmp * (np.identity(n) - tmp.T), dout)] 

# @custom_gradient(softmax_grad)
# def softmax(x):
#     x.unfreeze()
#     tmp = np.exp(x)
#     return tmp / np.sum(tmp) 


class Softmax(Layer):
    def __init__(self, name=None, **kwargs) -> None:
        super().__init__(name, **kwargs)

    def call(self, x):
        tmp = np.exp(x)
        return tmp / np.sum(tmp)
        # # * need to freeze the array to allow the whole array to enter the ufunc.
        # x.freeze()
        # return softmax([x])[0]