#  Copyright (c) 2022 Szymon Mikler

import torch
import torch.nn.functional as F

from pytorch_symbolic import Input, SymbolicModel, add_to_graph


def test_func_concat():
    in1 = Input((10, 20))
    in2 = Input((10, 20))

    cat1 = add_to_graph(torch.concat, (in1, in2), dim=2)
    cat2 = add_to_graph(torch.concat, tensors=(in1, in2), dim=2)

    model1 = SymbolicModel(inputs=(in1, in2), outputs=cat1)
    model2 = SymbolicModel(inputs=(in1, in2), outputs=cat2)

    for _ in range(10):
        x1 = torch.rand(1, 10, 20)
        x2 = torch.rand(1, 10, 20)
        correct_result = torch.concat((x1, x2), dim=2)
        assert torch.equal(model1(x1, x2), correct_result)
        assert torch.equal(model2(x1, x2), correct_result)


def test_detached_func_concat():
    in1 = Input((10, 20))
    in2 = Input((10, 20))

    cat1 = add_to_graph(torch.concat, (in1, in2), dim=2)
    cat2 = add_to_graph(torch.concat, tensors=(in1, in2), dim=2)

    model1 = SymbolicModel(inputs=(in1, in2), outputs=cat1).detach_from_graph()
    model2 = SymbolicModel(inputs=(in1, in2), outputs=cat2).detach_from_graph()

    for _ in range(10):
        x1 = torch.rand(1, 10, 20)
        x2 = torch.rand(1, 10, 20)
        correct_result = torch.concat((x1, x2), dim=2)
        assert torch.equal(model1(x1, x2), correct_result)
        assert torch.equal(model2(x1, x2), correct_result)


def sum_list_recursively(multiply_result, container):
    container = container.copy()
    for idx, el in enumerate(container):
        if isinstance(el, list):
            container[idx] = sum_list_recursively(1, el)
    return sum(container) * multiply_result


def test_func_complicated_input():
    sym1 = Input(batch_shape=(10, 20))
    sym2 = Input(batch_shape=(10, 20))
    sym3 = Input(batch_shape=(10, 20))

    other = torch.rand(10, 20)

    summed1 = add_to_graph(
        sum_list_recursively,
        5,
        [
            other,
            other,
            [other, other, sym1],
            [other, other, [other, other, [other]]],
            [other, sym2, other],
            [[[other], [[[[other, other]]], [other, [sym3, [other]]]]]],
        ],
    )
    summed2 = add_to_graph(
        sum_list_recursively, multiply_result=5, container=[[other] * 16, [sym1, [sym2, [sym3]]]]
    )

    model1 = SymbolicModel((sym1, sym2, sym3), outputs=summed1)
    model2 = SymbolicModel((sym1, sym2, sym3), outputs=summed2)

    x1 = torch.rand(10, 20)
    x2 = torch.rand(10, 20)
    x3 = torch.rand(10, 20)

    r1 = model1(x1, x2, x3)
    r2 = model2(x1, x2, x3)
    assert torch.allclose(r1, r2)

    correct_result = 5 * (other * 16 + x1 + x2 + x3)
    assert torch.allclose(r1, correct_result)


def test_detached_func_complicated_input():
    sym1 = Input(batch_shape=(10, 20))
    sym2 = Input(batch_shape=(10, 20))
    sym3 = Input(batch_shape=(10, 20))

    other = torch.rand(10, 20)

    summed1 = add_to_graph(
        sum_list_recursively,
        5,
        [
            other,
            other,
            [other, other, sym1],
            [other, other, [other, other, [other]]],
            [other, sym2, other],
            [[[other], [[[[other, other]]], [other, [sym3, [other]]]]]],
        ],
    )
    summed2 = add_to_graph(
        sum_list_recursively, multiply_result=5, container=[[other] * 16, [sym1, [sym2, [sym3]]]]
    )

    model1 = SymbolicModel((sym1, sym2, sym3), outputs=summed1).detach_from_graph()
    model2 = SymbolicModel((sym1, sym2, sym3), outputs=summed2).detach_from_graph()

    x1 = torch.rand(10, 20)
    x2 = torch.rand(10, 20)
    x3 = torch.rand(10, 20)

    r1 = model1(x1, x2, x3)
    r2 = model2(x1, x2, x3)
    assert torch.allclose(r1, r2)

    correct_result = 5 * (other * 16 + x1 + x2 + x3)
    assert torch.allclose(r1, correct_result)


def test_conv():
    inputs = Input(shape=(3, 32, 32))
    kernel = Input(batch_shape=(16, 3, 3, 3))
    bias = Input(batch_shape=(16,))
    output = add_to_graph(F.conv2d, input=inputs, weight=kernel, bias=bias, padding=1)
    model = SymbolicModel((inputs, kernel, bias), output)

    i = torch.rand(10, 3, 32, 32)
    k = torch.rand(16, 3, 3, 3)
    b = torch.rand(16)
    assert torch.allclose(model(i, k, b), F.conv2d(i, k, b, padding=1))
