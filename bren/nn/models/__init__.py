from bren.nn.models.Model import Model
from bren.nn.models.Sequential import Sequential
from bren.nn.utils import AliasDict
from bren.nn.layers import __all__ as layers
import pickle


MODELS = AliasDict({
    None: lambda: NameError("No such model found :'(")
})


def get_model(name): return MODELS[name]


def load_model(filepath, custom_objects={}):
    OBJECTS = {**custom_objects}
    for layer in layers: OBJECTS[layer.__name__] = layer

    with open(filepath, "rb") as f:
        model = pickle.load(f)

    for i, layer in enumerate(model.layers):
        L = OBJECTS[layer["layer"]](**layer)
        model.layers[i] = L

    model.assemble(**model.config)

    return model


__all__ = [Model, Sequential]

for cls in __all__:
    MODELS[cls.__name__] = cls