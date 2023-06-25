from bren.nn.models.Model import Model
from bren.nn.models.Sequential import Sequential
from bren.nn.utils import AliasDict
import pickle


MODELS = AliasDict()


def get_model(name): return MODELS[name]


def load_model(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    model = get_model(data["model"])([], params=data["trainable"])
    model.assemble(**data)

    return model.__dict__


__all__ = [Model, Sequential]

for cls in __all__:
    MODELS[cls.__name__] = cls