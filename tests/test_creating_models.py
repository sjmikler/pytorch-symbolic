#  Copyright (c) 2022 Szymon Mikler

import torch
from torch import nn

from pytorch_symbolic import Input, SymbolicModel, model_tools, optimize_module_calls, useful_layers


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
    x = inputs = Input(shape=(10,))
    torch.manual_seed(seed)

    for _ in range(5):
        x = x(nn.Linear(10, 10))
    return SymbolicModel(inputs, x)


def create_api_v2(seed):
    x = inputs = Input(batch_shape=(3, 10))
    torch.manual_seed(seed)

    for _ in range(5):
        x = nn.Linear(10, 10)(x)
    return SymbolicModel(inputs, x)


def test_equal_outputs():
    for seed in range(10):
        x = torch.rand(10, 10)
        outs0 = create_vanilla_pyt(seed)(x)
        outs1 = create_api_v1(seed)(x)
        outs2 = create_api_v2(seed)(x)
        assert torch.equal(outs1, outs2)
        assert torch.equal(outs1, outs0)


def test_equal_outputs_with_call_optimization():
    for seed in range(10):
        x = torch.rand(10, 10)
        model0 = create_vanilla_pyt(seed)
        model1 = create_api_v1(seed)
        model2 = create_api_v2(seed)
        optimize_module_calls()
        outs0 = model0(x)
        outs1 = model1(x)
        outs2 = model2(x)
        assert torch.equal(outs1, outs2)
        assert torch.equal(outs1, outs0)


def test_equal_parameters():
    for seed in range(10):
        model0 = create_vanilla_pyt(seed)
        model1 = create_api_v1(seed)
        model2 = create_api_v2(seed)
        assert model_tools.model_similar(model1, model2)
        assert model_tools.model_similar(model1, model0)
        assert model_tools.models_have_corresponding_parameters(model1, model2)
        assert model_tools.models_have_corresponding_parameters(model1, model0)


def create_vanilla_pyt_multi_in_out(seed):
    torch.manual_seed(seed)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.l11 = nn.Linear(10, 10)
            self.l12 = nn.Linear(10, 10)
            self.l21 = nn.Linear(10, 10)
            self.l22 = nn.Linear(10, 10)
            self.l00 = nn.Linear(20, 10)
            self.l13 = nn.Linear(10, 10)
            self.l23 = nn.Linear(10, 10)

        def forward(self, x1, x2):
            x1 = self.l11(x1)
            x1 = self.l12(x1)

            x2 = self.l21(x2)
            x2 = self.l22(x2)
            x_cat = torch.concat((x1, x2), dim=1)
            x_cat = self.l00(x_cat)

            x_out1 = self.l13(x_cat)
            x_out2 = self.l23(x_cat)
            return x_out1, x_out2

    return Model()


def create_api_multi_in_out(seed):
    inputs1 = x1 = Input(shape=(10,))
    inputs2 = x2 = Input(shape=(10,))
    torch.manual_seed(seed)

    for _ in range(2):
        x1 = nn.Linear(10, 10)(x1)

    for _ in range(2):
        x2 = nn.Linear(10, 10)(x2)

    x_cat = useful_layers.ConcatLayer(dim=1)(x1, x2)
    x_cat = nn.Linear(20, 10)(x_cat)

    x_out1 = nn.Linear(10, 10)(x_cat)
    x_out2 = nn.Linear(10, 10)(x_cat)
    return SymbolicModel((inputs1, inputs2), (x_out1, x_out2))


def test_equal_parameters_multi_in_out():
    for seed in range(10):
        model1 = create_vanilla_pyt_multi_in_out(seed)
        model2 = create_api_multi_in_out(seed)
        assert model_tools.model_similar(model1, model2)
        assert model_tools.models_have_corresponding_parameters(model1, model2)


def test_equal_outputs_multi_in_out():
    for seed in range(10):
        x1 = torch.rand(10, 10)
        x2 = torch.rand(10, 10)

        model1 = create_vanilla_pyt_multi_in_out(seed)
        o11, o12 = model1(x1, x2)

        model2 = create_api_multi_in_out(seed)
        o21, o22 = model2(x1, x2)

        assert torch.equal(o11, o21)
        assert torch.equal(o12, o22)


def test_detached_equal_parameters_multi_in_out():
    for seed in range(10):
        model1 = create_vanilla_pyt_multi_in_out(seed)
        model2 = create_api_multi_in_out(seed).detach_from_graph()
        assert model_tools.model_similar(model1, model2)
        assert model_tools.models_have_corresponding_parameters(model1, model2)


def test_detached_equal_outputs_multi_in_out():
    for seed in range(10):
        x1 = torch.rand(10, 10)
        x2 = torch.rand(10, 10)

        model1 = create_vanilla_pyt_multi_in_out(seed)
        o11, o12 = model1(x1, x2)

        model2 = create_api_multi_in_out(seed).detach_from_graph()
        o21, o22 = model2(x1, x2)

        assert torch.equal(o11, o21)
        assert torch.equal(o12, o22)


def test_creating_named_layers():
    inputs = Input((10,))
    layer = nn.Linear(10, 10)

    y1 = inputs(layer, custom_name="TEST_LAYER_777")
    y2 = layer(inputs, custom_name="TEST_LAYER_888")
    y = y1 + y2

    model = SymbolicModel(inputs, y)
    assert "TEST_LAYER_777" in str(model)
    assert "TEST_LAYER_888" in str(model)
