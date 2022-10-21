#  Copyright (c) 2022 Szymon Mikler

import torch
from torch import nn

from pytorch_symbolic import Input, SymbolicModel, model_tools


def AlwaysTheSameConv(in_channels, out_channels):
    torch.manual_seed(42)

    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


class VanillaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [AlwaysTheSameConv(3, 3) for _ in range(4)]
        for idx, layer in enumerate(self.layers):
            self.register_module(str(idx), layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_models_from_slice():
    x = Input(shape=(3, 10, 10))

    nodes = []
    for _ in range(20):
        x = AlwaysTheSameConv(x.C, 3)(x)
        unused = nn.Identity()(x)  # noqa: F841
        nodes.append(x)

    models = []
    for i in range(0, 20, 5):
        nodes_slice = nodes[i : i + 5]

        model = SymbolicModel(inputs=nodes_slice[0], outputs=nodes_slice[-1])
        models.append(model)

    data = torch.rand(16, 3, 10, 10)
    outs = prev_outs = models[-1](data)

    vanilla_model = VanillaModel()
    vanilla_outs = vanilla_model(data)
    assert torch.equal(outs, vanilla_outs)

    for model in models:
        outs = model(data)
        assert torch.equal(outs, prev_outs)
        assert model_tools.model_similar(model, vanilla_model)
        assert model_tools.models_have_corresponding_parameters(model, vanilla_model)
        prev_outs = outs


def test_detached_models_from_slice():
    x = Input(shape=(3, 10, 10))

    nodes = []
    for _ in range(20):
        x = AlwaysTheSameConv(x.C, 3)(x)
        unused = nn.Identity()(x)  # noqa: F841
        nodes.append(x)

    models = []
    for i in range(0, 20, 5):
        nodes_slice = nodes[i : i + 5]

        model = SymbolicModel(inputs=nodes_slice[0], outputs=nodes_slice[-1]).detach_from_graph()
        models.append(model)

    data = torch.rand(16, 3, 10, 10)
    outs = prev_outs = models[-1](data)

    vanilla_model = VanillaModel()
    vanilla_outs = vanilla_model(data)
    assert torch.equal(outs, vanilla_outs)

    for model in models:
        outs = model(data)
        assert torch.equal(outs, prev_outs)
        assert model_tools.model_similar(model, vanilla_model)
        assert model_tools.models_have_corresponding_parameters(model, vanilla_model)
        prev_outs = outs
