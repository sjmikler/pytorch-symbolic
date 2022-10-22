#  Copyright (c) 2022 Szymon Mikler

import random

import torch
from torch import nn

from pytorch_symbolic import Input, SymbolicModel, model_tools


def test_deep_linear():
    n_layers = 2500
    layers = [nn.Linear(4, 4) for _ in range(n_layers)]

    x = inputs = Input(shape=(4,))
    for layer in layers:
        x = layer(x)
    symbolic_model = SymbolicModel(inputs, x)
    model = nn.Sequential(*layers)

    x = torch.rand(size=(4, 4))
    assert torch.equal(model(x), symbolic_model(x))
    assert model_tools.model_similar(model, symbolic_model)
    assert model_tools.models_have_corresponding_parameters(model, symbolic_model)


def test_detached_deep_linear():
    n_layers = 2500
    layers = [nn.Linear(4, 4) for _ in range(n_layers)]

    x = inputs = Input(shape=(4,))
    for layer in layers:
        x = layer(x)
    symbolic_model = SymbolicModel(inputs, x).detach_from_graph()
    model = nn.Sequential(*layers)

    x = torch.rand(size=(4, 4))
    assert torch.equal(model(x), symbolic_model(x))
    assert model_tools.model_similar(model, symbolic_model)
    assert model_tools.models_have_corresponding_parameters(model, symbolic_model)


def test_random_models_from_large_graph():
    inputs = Input(shape=(4,))
    nodes = [inputs]

    n_layers = 2500
    for _ in range(n_layers):
        parent = random.choice(nodes)
        child = nn.Linear(4, 4)(parent)
        nodes.append(child)

    for _ in range(10):
        idx = random.randint(0, n_layers - 1)
        model = SymbolicModel(inputs=inputs, outputs=nodes[idx])
        _ = model(torch.rand(4, 4))

        counts = model_tools.get_parameter_count(model)
        num_layers = len(model._execution_order_layers)
        assert counts == num_layers * (16 + 4)
