#  Copyright (c) 2022 Szymon Mikler

import torch

from pytorch_symbolic import Input, SymbolicModel, optimize_module_calls


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
