#  Copyright (c) 2022 Szymon Mikler

import torch
from torch import nn

from pytorch_functional import FunctionalModel, Input, enable_module_call_optimization, tools


def create_api_v1(seed):
    torch.manual_seed(seed)
    x = inputs = Input(batch_shape=(10, 10))
    for _ in range(10):
        x = x(nn.Linear(x.features, x.features))
    x = x(nn.Linear(x.features, 5))
    return FunctionalModel(inputs, x)


def create_api_v2(seed):
    torch.manual_seed(seed)
    x = inputs = Input(batch_shape=(10, 10))
    for _ in range(10):
        x = nn.Linear(x.features, x.features)(x)
    x = nn.Linear(x.features, 5)(x)
    return FunctionalModel(inputs, x)


def test_equal_outputs():
    for seed in range(1, 10):
        x = torch.rand(10, 10)
        outs1 = create_api_v1(seed)(x)
        outs2 = create_api_v2(seed)(x)
        assert torch.equal(outs1, outs2)


def test_equal_outputs_with_call_optimization():
    enable_module_call_optimization()
    for seed in range(1, 10):
        x = torch.rand(10, 10)
        outs1 = create_api_v1(seed)(x)
        outs2 = create_api_v2(seed)(x)
        assert torch.equal(outs1, outs2)


def test_identical_networks():
    for seed in range(1, 10):
        x = torch.rand(10, 10)
        model1 = create_api_v1(seed)
        model2 = create_api_v2(seed)
        assert tools.model_identical(model1, model2)
