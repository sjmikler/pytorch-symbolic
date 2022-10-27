#  Copyright (c) 2022 Szymon Mikler

from __future__ import annotations

from typing import Callable, Dict, Hashable, List, Tuple

from torch import nn

from . import useful_layers
from .symbolic_data import SymbolicData


def add_module_to_graph(module, *args):
    assert isinstance(args[0], SymbolicData)
    return args[0](module, *args[1:])


def _replace_symbolic_with_value(container, extracted, navigation):
    """Search recursively for all occurences of Symbolic and replace them with their value.

    At the same time save navigation to know, how to do indexing to get to them.

    If navigation for container ends up as [..., [1, 2, 0, "TEST", 5], ...]
    this means that to get the element you should index container[1][2][0]["TEST"][5].
    """
    if isinstance(container, SymbolicData):
        navigation.append(navigation[-1].copy())
        extracted.append(container)
        return container.v

    if isinstance(container, List) or isinstance(container, Tuple):
        new_list = []
        for i, x in enumerate(container):
            navigation[-1].append(i)
            new_list.append(_replace_symbolic_with_value(x, extracted, navigation))
            navigation[-1].pop()
        return new_list
    if isinstance(container, Dict):
        new_dict = {}
        for k, v in container.items():
            navigation[-1].append(k)
            new_dict[k] = _replace_symbolic_with_value(v, extracted, navigation)
            navigation[-1].pop()
        return new_dict
    return container


def add_to_graph(func: Callable | nn.Module, *args, **kwds):
    """Register a custom func or module in the computation graph.

    This should work will arbitrary functions and modules.

    In case of functions it might add a small delay to the call, because it is figuring out
    where the arguments should go. If this is unacceptable, please create a nn.Module from your func.

    All arguments, including Symbolic Tensors, should be passed after the ``func`` argument.
    They can be mixed and matched, even nested in lists, tuples and dictionaries.

    Convolution func example::

        inputs = Input(shape=(3, 32, 32))
        kernel = Input(shape=(16, 3, 3, 3), batched=False)
        bias = Input(shape=(16,), batched=False)
        output = add_to_graph(F.conv2d, input=inputs, weight=k, bias=bias, padding=1)
    """
    if isinstance(func, nn.Module) and not kwds:
        return add_module_to_graph(func, *args)

    extracted_symbols: List[SymbolicData] = []

    real_call_args = []
    real_call_kwds = {}

    navigation: List[List[Hashable]] = [[]]
    real_call_args = _replace_symbolic_with_value(args, extracted_symbols, navigation)
    real_call_kwds = _replace_symbolic_with_value(kwds, extracted_symbols, navigation)
    navigation.pop()

    assert len(extracted_symbols) > 0, "No Symbolic Tensors detected in the input!"
    assert all((isinstance(symbol, SymbolicData) for symbol in extracted_symbols))
    assert len(extracted_symbols) == len(navigation)

    def wrapper_function(*args):
        assert len(args) == len(navigation), f"Expected {len(navigation)} inputs, not {len(args)}!"
        for arg, navi in zip(args, navigation):
            obj = real_call_kwds if isinstance(navi[0], str) else real_call_args

            for idx in navi[:-1]:
                obj = obj[idx]

            obj[navi[-1]] = arg

        return func(*real_call_args, **real_call_kwds)

    module = useful_layers.NamedAnyOpLayer(op=wrapper_function, name=f"{func.__name__}({len(navigation)})")
    return extracted_symbols[0](module, *extracted_symbols[1:])
