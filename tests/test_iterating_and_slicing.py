#  Copyright (c) 2022 Szymon Mikler

import torch
from torch import nn

from pytorch_symbolic import CustomInput, Input, SymbolicModel, add_to_graph


def test_iter_sum():
    x = Input(batch_shape=(10, 20))

    def tower(x):
        for _ in range(10):
            x = nn.Identity()(x)
        return x

    summed = y = Input(batch_shape=(20,))
    for i, row in enumerate(x):
        summed += tower(row * (i + 1))
    model1 = SymbolicModel(inputs=(x, y), outputs=summed)

    summed = y = Input(batch_shape=(20,))
    for i, row in enumerate(x):
        summed += row * (i + 1)
    model2 = SymbolicModel(inputs=(x, y), outputs=summed)

    torch.manual_seed(0)
    x = torch.rand(10, 20)
    real_result = torch.zeros(size=(20,))

    for i, row in enumerate(x):
        real_result += row * (i + 1)

    y = torch.zeros(size=(20,))
    our_result1 = model1(x, y)
    our_result2 = model2(x, y)
    assert torch.equal(real_result, our_result1)
    assert torch.equal(real_result, our_result2)


def test_indexing_basic1():
    x = Input((10,))

    def f(x):
        return x, [1, 2]

    outs1, outs2 = add_to_graph(f, x)
    _ = outs2[0]
    _ = outs2[1]


def test_unpacking_non_tensor():
    x = CustomInput(data=[5, 6, 7])

    model = SymbolicModel(inputs=x, outputs=(*x,))

    real_x = [9, 0, 1]
    outs = model(real_x)
    assert outs == (9, 0, 1)
