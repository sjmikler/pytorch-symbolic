#  Copyright (c) 2022 Szymon Mikler

import inspect
import logging

from torch import nn

from .functional_model import Placeholder

_objects_with_wrapped_call = []


def enable_functional_api_for_module(module):
    """Enable functional API.

    This means:
        * create __pytorch_functional_old_call__ that holds old call
        * wrap __call__
    """
    assert inspect.isclass(module), "Argument must be a class!"
    logging.debug(f"ENABLING EXPERIMENTAL API FOR {module}")

    assert "__pytorch_functional_old_call__" not in vars(
        module
    ), f"Functional API already enabled for {module}!"

    __old_call__ = module.__call__
    module.__pytorch_functional_old_call__ = __old_call__

    def experimental_monkey_patch_call(self, *args, **kwds):
        if len(args) > 0 and len(kwds) == 0 and all((isinstance(x, Placeholder) for x in args)):
            node = args[0]
            return node.apply_layer(self, *args[1:])
        else:
            return __old_call__(self, *args, **kwds)

    _objects_with_wrapped_call.append(module)
    module.__call__ = experimental_monkey_patch_call


def disable_functional_api_for_module(module):
    logging.debug(f"DISABLING EXPERIMENTAL API FOR {module}")

    assert hasattr(module, "__pytorch_functional_old_call__"), f"Functional API not enabled for {module}!"
    module.__call__ = module.__pytorch_functional_old_call__
    del module.__pytorch_functional_old_call__


def functional_api_new_wrapper(self, *args, **kwds):
    obj = super(nn.Module, self).__new__(self)
    if type(obj) not in _objects_with_wrapped_call:
        enable_functional_api_for_module(type(obj))
    return obj


def functional_api_new_wrapper_backup(self, *args, **kwds):
    obj = super(nn.Module, self).__new__(self)
    return obj


def enable_functional_api_for_new_modules():
    nn.Module.__new__ = functional_api_new_wrapper


def disable_functional_api_for_new_modules():
    nn.Module.__new__ = functional_api_new_wrapper_backup


def remove_call_wrapper_from_all_modules():
    global _objects_with_wrapped_call
    for obj in _objects_with_wrapped_call:
        disable_functional_api_for_module(obj)
    _objects_with_wrapped_call = []
