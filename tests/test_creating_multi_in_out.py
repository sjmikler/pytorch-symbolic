#  Copyright (c) 2022 Szymon Mikler

import torch
from torch import nn

from pytorch_symbolic import Input, SymbolicModel, model_tools

NUM_INPUTS = 17
NUM_LAYERS = 11
FEATURES = 7
SEED = 2137


class WeightedConcatLayer(nn.Module):
    def __init__(self, dim):
        """Strange layer with multiple inputs where their correct order is important."""
        super().__init__()
        self.dim = dim

    def forward(self, *tensors):
        tensors = [tensor * (i + 1) for i, tensor in enumerate(tensors)]
        return torch.cat(tensors=tensors, dim=self.dim)


class VanillaModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = [nn.Linear(NUM_INPUTS * FEATURES, FEATURES) for _ in range(NUM_LAYERS)]
        for idx, layer in enumerate(self.layers):
            self.register_module(str(idx), layer)

        self.wconcat = WeightedConcatLayer(dim=1)
        self.final_linear = nn.Linear(NUM_LAYERS * FEATURES, FEATURES)
        self.flatten = nn.Flatten()

    def forward(self, *inputs):
        concatenated = self.wconcat(*inputs)
        transformed = [layer(concatenated) for layer in self.layers]
        concatenated2 = self.wconcat(*transformed)
        concatenated2 = self.final_linear(concatenated2)
        return concatenated2 - sum(transformed)


def test_multiple_inputs():
    inputs = [Input(shape=(FEATURES,)) for _ in range(NUM_INPUTS)]
    torch.manual_seed(SEED)

    concatenated = WeightedConcatLayer(dim=1)(*inputs)
    transformed = [nn.Linear(concatenated.features, FEATURES)(concatenated) for _ in range(NUM_LAYERS)]
    concatenated2 = WeightedConcatLayer(dim=1)(*transformed)
    concatenated2 = nn.Linear(concatenated2.features, FEATURES)(concatenated2)
    result = concatenated2 - sum(transformed)
    model = SymbolicModel(inputs=inputs, outputs=result)

    torch.manual_seed(SEED)
    vanilla_model = VanillaModel()

    assert model_tools.model_similar(model, vanilla_model)
    assert model_tools.models_have_corresponding_parameters(model, vanilla_model)

    for i in range(10):
        xs = [torch.rand(i, FEATURES) for _ in range(NUM_INPUTS)]
        out = model(*xs)
        vanilla_out = vanilla_model(*xs)
        assert torch.equal(out, vanilla_out), str((out, vanilla_out))


def test_detached_multiple_inputs():
    inputs = [Input(shape=(FEATURES,)) for _ in range(NUM_INPUTS)]
    torch.manual_seed(SEED)

    concatenated = WeightedConcatLayer(dim=1)(*inputs)
    transformed = [nn.Linear(concatenated.features, FEATURES)(concatenated) for _ in range(NUM_LAYERS)]
    concatenated2 = WeightedConcatLayer(dim=1)(*transformed)
    concatenated2 = nn.Linear(concatenated2.features, FEATURES)(concatenated2)
    result = concatenated2 - sum(transformed)
    model = SymbolicModel(inputs=inputs, outputs=result).detach_from_graph()

    torch.manual_seed(SEED)
    vanilla_model = VanillaModel()

    assert model_tools.model_similar(model, vanilla_model)
    assert model_tools.models_have_corresponding_parameters(model, vanilla_model)

    for i in range(10):
        xs = [torch.rand(i, FEATURES) for _ in range(NUM_INPUTS)]
        out = model(*xs)
        vanilla_out = vanilla_model(*xs)
        assert torch.equal(out, vanilla_out), str((out, vanilla_out))
