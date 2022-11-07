#  Copyright (c) 2022 Szymon Mikler

import inspect
import logging

from torch import nn

from .symbolic_data import SymbolicData

_objects_with_wrapped_call = []


def enable_symbolic_API_for_module(module):
    """Enable symbolic API.

    This means:
        * create __pytorch_symbolic_old_call__ that holds old call in case we want to restore it
        * wrap __call__ to check whether the inputs are Symbolic Data
    """
    assert inspect.isclass(module), "Argument must be a class!"
    logging.debug(f"ENABLING SYMBOLIC API FOR {module}")

    assert "__pytorch_symbolic_old_call__" not in vars(module), f"Symbolic API already enabled for {module}!"

    __old_call__ = module.__call__
    module.__pytorch_symbolic_old_call__ = __old_call__

    def experimental_monkey_patch_call(self, *args, **kwds):
        if not any(isinstance(x, SymbolicData) for x in args):
            return __old_call__(self, *args, **kwds)
        elif len(kwds) == 0 and all(isinstance(x, SymbolicData) for x in args):
            node = args[0]
            return node(self, *args[1:])
        else:
            msg = (
                "Only unnamed SymbolicData are allowed as arguments! "
                "If you need more flexibility, use `functions_utility.add_to_graph`!"
            )
            raise UserWarning(msg)

    _objects_with_wrapped_call.append(module)
    module.__call__ = experimental_monkey_patch_call


def disable_symbolic_API_for_module(module):
    logging.debug(f"DISABLING SYMBOLIC API FOR {module}")

    assert hasattr(module, "__pytorch_symbolic_old_call__"), f"Symbolic API not enabled for {module}!"
    module.__call__ = module.__pytorch_symbolic_old_call__
    del module.__pytorch_symbolic_old_call__


def symbolic_API__new__wrapper(self, *args, **kwds):
    obj = super(nn.Module, self).__new__(self)
    if type(obj) not in _objects_with_wrapped_call:
        enable_symbolic_API_for_module(type(obj))
    return obj


def symbolic_API__new_wrapper_backup(self, *args, **kwds):
    obj = super(nn.Module, self).__new__(self)
    return obj


def enable_symbolic_API_for_new_modules():
    nn.Module.__new__ = symbolic_API__new__wrapper


def disable_symbolic_api_for_new_modules():
    nn.Module.__new__ = symbolic_API__new_wrapper_backup


def remove_call_wrapper_from_all_modules():
    global _objects_with_wrapped_call
    for obj in _objects_with_wrapped_call:
        disable_symbolic_API_for_module(obj)
    _objects_with_wrapped_call = []
