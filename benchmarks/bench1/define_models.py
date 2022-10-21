#  Copyright (c) 2022 Szymon Mikler

from torch import nn

from pytorch_symbolic import Input, SymbolicModel


class Vanilla(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        for idx, layer in enumerate(layers):
            self.add_module(str(idx), layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def create_sequential_multi_layer(n_layers, n_features):
    """Very thin, unrealistic network whose bootleneck is the number of subsequent calls."""
    layers = [nn.Linear(n_features, n_features) for _ in range(n_layers)]

    x = inputs = Input(shape=(n_features,))
    for layer in layers:
        x = x(layer)

    models = [
        (("functional",), SymbolicModel(inputs, x)),
        (("sequential",), nn.Sequential(*layers)),
        (("vanilla",), Vanilla(layers)),
    ]
    return models
