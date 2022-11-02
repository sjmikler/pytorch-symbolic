#  Copyright (c) 2022 Szymon Mikler

import torch
from torch import nn

from pytorch_symbolic import CustomInput, Input, SymbolicModel, add_to_graph, optimize_module_calls


def test_all():
    x1 = Input((5, 5))
    x2 = Input((5, 5))
    const = torch.rand(5, 5)

    results1 = [
        abs(x1),
        abs(x2),
        x1 + x2,
        x1 - x2 / 2,
        x1 * x2,
        x1 / x2,
        x1 % x2,
        x1 @ x2,
        x1**x2,
    ]
    model1 = SymbolicModel(inputs=(x1, x2), outputs=sum(results1))

    results2 = [
        x1 + const,
        x1 - const / 2,
        x1 * const,
        x1 / const,
        x1 % const,
        x1 @ const,
        x1**const,
    ]
    model2 = SymbolicModel(inputs=(x1, x2), outputs=sum(results2))

    results3 = [
        const + x2,
        const - x2 / 2,
        const * x2,
        const / x2,
        const % x2,
        const @ x2,
        const**x2,
    ]
    model3 = SymbolicModel(inputs=(x1, x2), outputs=sum(results3))

    for _ in range(10):
        x1 = torch.rand(5, 5)
        x2 = torch.rand(5, 5)
        out1 = model1(x1, x2)
        res1 = abs(x1) + abs(x2) + x1 + x2 + x1 - x2 / 2 + x1 * x2 + x1 / x2 + x1 % x2 + x1 @ x2 + x1**x2
        assert torch.allclose(out1, res1)

        out2 = model2(x1, x2)
        res2 = x1 + const + x1 - const / 2 + x1 * const + x1 / const + x1 % const + x1 @ const + x1**const
        assert torch.allclose(out2, res2)

        out3 = model3(x1, x2)
        res3 = const + x2 + const - x2 / 2 + const * x2 + const / x2 + const % x2 + const @ x2 + const**x2
        assert torch.allclose(out3, res3)


def test_all_detached():
    x1 = Input((5, 5))
    x2 = Input((5, 5))
    const = torch.rand(5, 5)

    results1 = [
        abs(x1),
        abs(x2),
        x1 + x2,
        x1 - x2 / 2,
        x1 * x2,
        x1 / x2,
        x1 % x2,
        x1 @ x2,
        x1**x2,
    ]
    model1 = SymbolicModel(inputs=(x1, x2), outputs=sum(results1)).detach_from_graph()

    results2 = [
        x1 + const,
        x1 - const / 2,
        x1 * const,
        x1 / const,
        x1 % const,
        x1 @ const,
        x1**const,
    ]
    model2 = SymbolicModel(inputs=(x1, x2), outputs=sum(results2)).detach_from_graph()

    results3 = [
        const + x2,
        const - x2 / 2,
        const * x2,
        const / x2,
        const % x2,
        const @ x2,
        const**x2,
    ]
    model3 = SymbolicModel(inputs=(x1, x2), outputs=sum(results3)).detach_from_graph()

    for _ in range(10):
        x1 = torch.rand(5, 5)
        x2 = torch.rand(5, 5)
        out1 = model1(x1, x2)
        res1 = abs(x1) + abs(x2) + x1 + x2 + x1 - x2 / 2 + x1 * x2 + x1 / x2 + x1 % x2 + x1 @ x2 + x1**x2
        assert torch.allclose(out1, res1)

        out2 = model2(x1, x2)
        res2 = x1 + const + x1 - const / 2 + x1 * const + x1 / const + x1 % const + x1 @ const + x1**const
        assert torch.allclose(out2, res2)

        out3 = model3(x1, x2)
        res3 = const + x2 + const - x2 / 2 + const * x2 + const / x2 + const % x2 + const @ x2 + const**x2
        assert torch.allclose(out3, res3)


def test_all_optimized():
    x1 = Input((5, 5))
    x2 = Input((5, 5))
    const = torch.rand(5, 5)

    results1 = [
        abs(x1),
        abs(x2),
        x1 + x2,
        x1 - x2 / 2,
        x1 * x2,
        x1 / x2,
        x1 % x2,
        x1 @ x2,
        x1**x2,
    ]
    model1 = SymbolicModel(inputs=(x1, x2), outputs=sum(results1))

    results2 = [
        x1 + const,
        x1 - const / 2,
        x1 * const,
        x1 / const,
        x1 % const,
        x1 @ const,
        x1**const,
    ]
    model2 = SymbolicModel(inputs=(x1, x2), outputs=sum(results2))

    results3 = [
        const + x2,
        const - x2 / 2,
        const * x2,
        const / x2,
        const % x2,
        const @ x2,
        const**x2,
    ]
    model3 = SymbolicModel(inputs=(x1, x2), outputs=sum(results3))

    optimize_module_calls()

    for _ in range(10):
        x1 = torch.rand(5, 5)
        x2 = torch.rand(5, 5)
        out1 = model1(x1, x2)
        res1 = abs(x1) + abs(x2) + x1 + x2 + x1 - x2 / 2 + x1 * x2 + x1 / x2 + x1 % x2 + x1 @ x2 + x1**x2
        assert torch.allclose(out1, res1)

        out2 = model2(x1, x2)
        res2 = x1 + const + x1 - const / 2 + x1 * const + x1 / const + x1 % const + x1 @ const + x1**const
        assert torch.allclose(out2, res2)

        out3 = model3(x1, x2)
        res3 = const + x2 + const - x2 / 2 + const * x2 + const / x2 + const % x2 + const @ x2 + const**x2
        assert torch.allclose(out3, res3)


def test_mmul_on_list_torch():
    A, B, C = 5, 5, 5
    x = CustomInput(data=[[torch.rand(1) for _ in range(B)] for _ in range(A)])
    y = CustomInput(data=[[torch.rand(1) for _ in range(C)] for _ in range(B)])

    r = None
    for i in range(len(x[0])):
        for j in range(len(y)):

            class Mmul(nn.Module):
                def __init__(self, i, j):
                    super().__init__()
                    self.i = i
                    self.j = j

                def forward(self, x, y):
                    r = torch.zeros(len(x), len(y[0]))
                    assert len(x[0]) == len(y), "Wrong shapes!"
                    r[self.i][self.j] = sum([x[self.i][p] * y[p][self.j] for p in range(len(y))])
                    return r

            mmul = Mmul(i, j)

            if r is None:
                r = mmul(x, y)
            else:
                r = r + mmul(x, y)
    model = SymbolicModel(inputs=(x, y), outputs=(r,))

    X = torch.rand(A, B)
    Y = torch.rand(B, C)

    R = model(X.tolist(), Y.tolist())
    assert torch.allclose(R, X @ Y)


def test_mmul_on_list_numpy():
    import numpy as np

    A, B, C = 5, 5, 5
    x = CustomInput(data=[[np.random.rand() for _ in range(B)] for _ in range(A)])
    y = CustomInput(data=[[np.random.rand() for _ in range(C)] for _ in range(B)])

    rs = []
    for i in range(len(x[0])):
        for j in range(len(y)):

            class Mmul(nn.Module):
                def __init__(self, i, j):
                    super().__init__()
                    self.i = i
                    self.j = j

                def forward(self, x, y):
                    r = np.zeros([len(x), len(y[0])])
                    assert len(x[0]) == len(y), "Wrong shapes!"
                    r[self.i][self.j] = sum([x[self.i][p] * y[p][self.j] for p in range(len(y))])
                    return r

            mmul = Mmul(i, j)
            r = mmul(x, y)
            rs.append(r)

    output = add_to_graph(lambda *args: sum(args), *rs)
    model = SymbolicModel(inputs=(x, y), outputs=output)

    X = np.random.rand(A, B)
    Y = np.random.rand(B, C)

    R = model(X.tolist(), Y.tolist())
    assert np.allclose(R, X @ Y)


def test_anypow_layer():
    tensor = Input(shape=(10, 20))
    power = CustomInput(data=1.5)

    class AnyPow(nn.Module):
        def forward(self, tensor, power):
            return tensor**power

    output = AnyPow()(tensor, power)

    model = SymbolicModel(inputs=(tensor, power), outputs=output)

    for x in range(10):
        for y in [0.5, 1.0, 1.5, 2.0]:
            assert model(x, y) == x**y

    x = torch.rand(10, 20, 30)
    for y in [0.5, 1.0, 1.5, 2.0]:
        assert torch.allclose(model(x, y), x**y)
        assert torch.allclose(model.detach_from_graph()(x, y), x**y)

    x = torch.rand(10, 20, 30)
    for y in [0.5, 1.0, 1.5, 2.0]:
        assert torch.allclose(model(y, x), y**x)
        assert torch.allclose(model.detach_from_graph()(y, x), y**x)


def test_indexing_symbolic_data():
    inputs = CustomInput(
        {
            "x": torch.rand(10, 20),
            "y": torch.rand(10, 20),
        }
    )
    key = CustomInput(data="x")
    value = inputs[key]
    outputs = nn.Identity()(value)

    model = SymbolicModel(inputs=(inputs, key), outputs=outputs)

    real_inputs = {
        "x": torch.rand(1, 2, 3),
        "y": torch.rand(1, 2, 3),
        "z": torch.rand(1, 2, 3),
    }

    for key in real_inputs:
        outs = model(real_inputs, key)
        assert torch.equal(outs, real_inputs[key])
