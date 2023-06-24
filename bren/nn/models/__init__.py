from bren.nn.models.Model import Model
from bren.nn.models.Sequential import Sequential
import pickle


def load_model(filepath):
    with open(filepath, "rb") as f:
        weights = pickle.load(f)



__all__ = [Model, Sequential]