#  Copyright (c) 2022 Szymon Mikler

from torch import nn

from pytorch_symbolic import Input, SymbolicModel


def test_cycle1():
    x1 = nn.Linear(2, 2)(Input(shape=(2,)))
    x2 = nn.Linear(2, 2)(x1)
    x3 = nn.Linear(2, 2)(x2)
    x4 = nn.Linear(2, 2)(x3)
    x5 = nn.Linear(2, 2)(x4)

    x5._children.append(x1)
    x1._parents = (x5,)

    try:
        _ = SymbolicModel(inputs=x2, outputs=x4)
        raise UserWarning("Created model with a cycle! Assertion should have been raised!")
    except AssertionError:
        pass


def test_cycle2():
    x1 = nn.Linear(2, 2)(Input(shape=(2,)))
    x2 = nn.Linear(2, 2)(x1)
    x3 = nn.Linear(2, 2)(x2)
    x4 = x3 + Input(shape=(2,))
    x5 = nn.Linear(2, 2)(x4)
    x6 = nn.Linear(2, 2)(x5)
    x7 = nn.Linear(2, 2)(x6)

    x4._parents = (x3, x5)  # introduce a cycle

    try:
        _ = SymbolicModel(inputs=x2, outputs=x7)
        raise UserWarning("Created model with a cycle! Assertion should have been raised!")
    except AssertionError:
        pass
