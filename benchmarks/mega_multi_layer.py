#  Copyright (c) 2022 Szymon Mikler

import logging
import time

import torch
from torch import nn

from pytorch_functional import FunctionalModel, Input, optimize_module_calls

N_WARMUP = 10
N_ITER = 1000
N_LAYERS = [10, 20, 40, 80, 160, 320, 640, 1280]
N_FEATURES = 4
BATCH_SIZE = 4


class VanillaMultiLayerNet(nn.Module):
    def __init__(self, n_layers):
        super().__init__()

        self.layers = []
        for i in range(n_layers):
            layer = nn.Linear(N_FEATURES, N_FEATURES)
            self.register_module(str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def mega_multi_layer_net(n_layers, cuda_graphs=False, code_gen=True):
    """Very thin, unrealistic network whose bootleneck is the number of subsequent calls."""
    x = inputs = Input(batch_shape=(BATCH_SIZE, N_FEATURES))

    for _ in range(n_layers):
        x = nn.Linear(x.features, x.features)(x)

    model = FunctionalModel(inputs, x, enable_cuda_graphs=cuda_graphs, generate_optimized_forward=code_gen)
    return model


def run(model):
    x = torch.rand(size=(BATCH_SIZE, N_FEATURES)).cpu()

    for _ in range(N_WARMUP):
        outputs = model(x)
    torch.cuda.synchronize()

    # TIMING

    t0 = time.time()
    for _ in range(N_ITER):
        outputs = model(x)
    torch.cuda.synchronize()
    td = time.time() - t0
    return td


if __name__ == "__main__":
    assert torch.cuda.is_available()

    torch.cuda.set_device(1)

    import sys

    sys.setrecursionlimit(5000)

    # Create the network

    models = {n_layers: mega_multi_layer_net(n_layers).cpu() for n_layers in N_LAYERS}
    models_va = {n_layers: VanillaMultiLayerNet(n_layers).cpu() for n_layers in N_LAYERS}
    models_cg = {n_layers: mega_multi_layer_net(n_layers, cuda_graphs=True).cpu() for n_layers in N_LAYERS}

    for n_layers, model in models.items():
        td = run(model)
        time_per_run = td / N_ITER
        print(f"layers: {n_layers: > 5}, MS per run: {time_per_run * 100:6.4f}")
    print()

    for n_layers, model in models_va.items():
        td = run(model)
        time_per_run = td / N_ITER
        print(f"layers: {n_layers: > 5}, MS per run: {time_per_run * 100:6.4f}")
    print()

    for n_layers, model in models_cg.items():
        td = run(model)
        time_per_run = td / N_ITER
        print(f"layers: {n_layers: > 5}, MS per run: {time_per_run * 100:6.4f}")
    print()

    optimize_module_calls()
    time.sleep(0.1)

    for n_layers, model in models.items():
        td = run(model)
        time_per_run = td / N_ITER
        print(f"layers: {n_layers: > 5}, MS per run: {time_per_run * 100:6.4f}")
    print()

    for n_layers, model in models_va.items():
        td = run(model)
        time_per_run = td / N_ITER
        print(f"layers: {n_layers: > 5}, MS per run: {time_per_run * 100:6.4f}")
    print()

    for n_layers, model in models_cg.items():
        td = run(model)
        time_per_run = td / N_ITER
        print(f"layers: {n_layers: > 5}, MS per run: {time_per_run * 100:6.4f}")
    print()
