#  Copyright (c) 2022 Szymon Mikler

import inspect
import logging

from torch import nn

from . import layers
from .functional_model import Placeholder

logging.debug("STARTING DETECTING MODULES")

predefined_modules = []
for value in nn.__dict__.values():
    if inspect.isclass(value) and issubclass(value, nn.Module):
        logging.debug(f"DETECTED {value}")
        predefined_modules.append(value)
for value in layers.__dict__.values():
    if inspect.isclass(value) and issubclass(value, nn.Module):
        logging.debug(f"DETECTED {value}")
        predefined_modules.append(value)

logging.debug("DETECTING MODULES FINISHED")


def enable_functional_api_for_package(pkg):
    """Search through the package and enable functional API for every nn.Module."""
    for value in pkg.__dict__.values():
        if inspect.isclass(value) and issubclass(value, nn.Module):
            logging.debug(f"DETECTED {value}")
            enable_functional_api_for_module(value)


def disable_functional_api_for_package(pkg):
    """Search through the package and disable functional API for every nn.Module."""
    for value in pkg.__dict__.values():
        if inspect.isclass(value) and issubclass(value, nn.Module):
            logging.debug(f"DETECTED {value}")
            disable_functional_api_for_module(value)


def enable_functional_api_for_module(module):
    """Enable functional API.

    This means:
        * create __pytorch_functional_old_call__ that holds old call
        * wrap __call__
    """
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

    module.__call__ = experimental_monkey_patch_call


def disable_functional_api_for_module(module):
    logging.debug(f"DISABLING EXPERIMENTAL API FOR {module}")

    assert hasattr(module, "__pytorch_functional_old_call__"), f"Functional API not enabled for {module}!"
    module.__call__ = module.__pytorch_functional_old_call__
    del module.__pytorch_functional_old_call__


def enable_experimental_api_for_predefined_modules():
    for module in predefined_modules:
        enable_functional_api_for_module(module)


def disable_experimental_api_for_predefined_modules():
    for module in predefined_modules:
        disable_functional_api_for_module(module)


def enable_experimental_api():
    enable_experimental_api_for_predefined_modules()
