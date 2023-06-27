from bren.nn.models import Model
import numpy as np
import bren as br


class Sequential(Model):
    def __init__(self, layers, **kwargs) -> None:
        super().__init__(**kwargs)
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
    
    def save(self, filepath):
        # for layer in self.layers:
        #     print(layer)
        #     if layer.__class__ not in br.nn.__all__: raise ValueError("custom layers cannot be saved.")
        # self.add_config("layers", self.layers)
        return super().save(filepath)