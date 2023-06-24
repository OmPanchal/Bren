from bren.nn.models import Model
import numpy as np


class Sequential(Model):
    def __init__(self, layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = layers

    def build(self, input):
        super().build(input)
        
        for layer in self.layers:
            self.add_weight(layer.trainable)
            # ^ have to add the trainable variables to the model so they can be updated..


    def call(self, x, training=None):
        Z = x
        for layer in self.layers:
            Z = layer(Z, training=training)
        return Z