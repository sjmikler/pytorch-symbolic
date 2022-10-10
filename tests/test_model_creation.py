#  Copyright (c) 2022 Szymon Mikler

import torch
from torch import nn

from pytorch_functional import FunctionalModel, Input, enable_module_call_optimization, tools


def create_vanilla_pyt(seed):
    torch.manual_seed(seed)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()

            self.layers = []
            for _ in range(5):
                self.layers.append(nn.Linear(10, 10))

            for idx, layer in enumerate(self.layers):
                self.add_module(name=str(idx), module=layer)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return Model()


def create_api_v1(seed):
    x = inputs = Input(batch_shape=(10, 10))
    torch.manual_seed(seed)

    for _ in range(5):
        x = x(nn.Linear(10, 10))
    return FunctionalModel(inputs, x)


def create_api_v2(seed):
    x = inputs = Input(batch_shape=(10, 10))
    torch.manual_seed(seed)

    for _ in range(5):
        x = nn.Linear(10, 10)(x)
    return FunctionalModel(inputs, x)


def test_equal_outputs():
    for seed in range(1, 10):
        x = torch.rand(10, 10)
        outs0 = create_vanilla_pyt(seed)(x)
        outs1 = create_api_v1(seed)(x)
        outs2 = create_api_v2(seed)(x)
        assert torch.equal(outs1, outs2)
        assert torch.equal(outs1, outs0)


def test_equal_outputs_with_call_optimization():
    enable_module_call_optimization()
    for seed in range(1, 10):
        x = torch.rand(10, 10)
        outs0 = create_vanilla_pyt(seed)(x)
        outs1 = create_api_v1(seed)(x)
        outs2 = create_api_v2(seed)(x)
        assert torch.equal(outs1, outs2)
        assert torch.equal(outs1, outs0)


def test_identical_networks():
    for seed in range(1, 10):
        model0 = create_vanilla_pyt(seed)
        model1 = create_api_v1(seed)
        model2 = create_api_v2(seed)
        assert tools.models_have_corresponding_parameters(model1, model2)
        assert tools.models_have_corresponding_parameters(model1, model0)
