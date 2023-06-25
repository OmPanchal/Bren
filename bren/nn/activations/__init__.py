from bren.nn.activations.Tanh import Tanh, tanh
from bren.nn.activations.ReLU import ReLU
from bren.nn.activations.ELU import ELU
from bren.nn.activations.Linear import Linear
from bren.nn.utils import AliasDict


__all__ = [Tanh, ReLU, ELU]


ACTIVATIONS = AliasDict({ None: Linear })

for cls in __all__:
	ACTIVATIONS[cls.__name__] = cls

ACTIVATIONS.add(ReLU.__name__, "relu")
ACTIVATIONS.add(ELU.__name__, "elu")
ACTIVATIONS.add(Tanh.__name__, "tanh")

def get_activation(name):
	return ACTIVATIONS[name]