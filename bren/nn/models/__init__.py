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
        # print(model.layers)

    model_layers = model.layers

    for i, layer in enumerate(model.layers):
        L = OBJECTS[layer["layer"]](**layer, custom_obs=custom_objects, params=layer.get("trainable", []))
        # print(L.activation)
        model_layers[i] = L
        model_layers[i].set_built(True)
        try: model_layers[i].activation.set_built(True)
        except AttributeError: pass 
        model_layers[i].__dict__ = {**model_layers[i].__dict__, **layer["trainable"]} 
        # print("\n\n", L.__dict__)

    # for layer in model.layers:
    #     print(layer.built)
    model.assemble(**model.config)
    
    return model


__all__ = [Model, Sequential]

for cls in __all__:
    MODELS[cls.__name__] = cls