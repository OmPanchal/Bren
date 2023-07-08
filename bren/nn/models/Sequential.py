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
            # print("HERE", layer.trainable)
            self.add_weight(layer.trainable)
            # ^ have to add the trainable variables to the model so they can be updated..

    def call(self, x, training=None):
        # print("Calling")
        Z = x
        for layer in self.layers:
            # print(layer.built)
            Z = layer(Z, training=training)
        return Z
    
    def save(self, filepath):
        # Serialise the layers before doing the standard save....
        for i, layer in enumerate(self.layers):
            self.layers[i] = layer.config()

        return super().save(filepath)