#  Copyright (c) 2022 Szymon Mikler

import random

from pytorch_symbolic import CustomInput, SymbolicModel


def test_set_opts():
    x = CustomInput(set())
    y = CustomInput(set())
    z = CustomInput(set())
    u = {i * 2 for i in range(50)}  # constant value

    outs = ((x & y ^ z) | (x ^ z - y)) & u
    model = SymbolicModel((x, y, z), outs)

    for case in range(10):
        x = set(random.randint(0, 100) for _ in range(20))
        y = set(random.randint(0, 100) for _ in range(20))
        z = set(random.randint(0, 100) for _ in range(20))

        outs = model(x, y, z)
        real_outs = ((x & y ^ z) | (x ^ z - y)) & u
        assert outs == real_outs


def test_illegal_inplace():
    """This test performs in-place operations which are discuraged!

    A lot might break when using them. Even if they output None, they are required in the outputs.
    If they won't be here, they won't be replayed. Even if they are there, other problems might happend
    when multiple models share the same nodes.
    """
    x = CustomInput([])
    o1 = x.append(1)
    o2 = x.append(2)
    o3 = x.append(3)
    model = SymbolicModel(inputs=x, outputs=(x, o1, o2, o3))

    outs, _, _, _ = model([3, 4, 5])
    assert outs == [3, 4, 5, 1, 2, 3]


def test_int_and_constants():
    x = CustomInput(5)
    y = CustomInput(1)
    z = 14

    a = x + 14
    b = z + x
    c = x + y

    model = SymbolicModel((x, y), outputs=(a, b, c))

    outs = model(22, 21)
    assert outs == (36, 36, 43)
