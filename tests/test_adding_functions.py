#  Copyright (c) 2022 Szymon Mikler

import torch

from pytorch_functional import FunctionalModel, Input, functions_utility


def test_func_concat():
    in1 = Input((10, 20))
    in2 = Input((10, 20))

    cat1 = functions_utility.add_to_model(torch.concat, (in1, in2), dim=2)
    cat2 = functions_utility.add_to_model(torch.concat, tensors=(in1, in2), dim=2)

    model1 = FunctionalModel(inputs=(in1, in2), outputs=cat1)
    model2 = FunctionalModel(inputs=(in1, in2), outputs=cat2)

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

    summed1 = functions_utility.add_to_model(
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
    summed2 = functions_utility.add_to_model(
        sum_list_recursively, multiply_result=5, container=[[other] * 16, [sym1, [sym2, [sym3]]]]
    )

    model1 = FunctionalModel((sym1, sym2, sym3), outputs=summed1)
    model2 = FunctionalModel((sym1, sym2, sym3), outputs=summed2)

    x1 = torch.rand(10, 20)
    x2 = torch.rand(10, 20)
    x3 = torch.rand(10, 20)

    r1 = model1(x1, x2, x3)
    r2 = model2(x1, x2, x3)
    assert torch.allclose(r1, r2)

    correct_result = 5 * (other * 16 + x1 + x2 + x3)
    assert torch.allclose(r1, correct_result)
