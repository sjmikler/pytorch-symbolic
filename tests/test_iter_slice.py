#  Copyright (c) 2022 Szymon Mikler

import torch
from torch import nn

from pytorch_symbolic import Input, SymbolicModel


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
