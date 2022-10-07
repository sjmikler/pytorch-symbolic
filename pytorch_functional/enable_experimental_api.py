import inspect
import logging

from torch import nn

from . import layers
from .functional_model import FMGraphNode

logging.debug("STARTING DETECTING MODULES")

predefined_modules = []
for value in nn.__dict__.values():
    if inspect.isclass(value) and issubclass(value, nn.Module):
        logging.debug(value)
        predefined_modules.append(value)
for value in layers.__dict__.values():
    if inspect.isclass(value) and issubclass(value, nn.Module):
        logging.debug(value)
        predefined_modules.append(value)

logging.debug("DETECTING MODULES FINISHED")

for module in predefined_modules:

    def replace_call(module):
        __old_call__ = module.__call__
        module.__old_call__ = __old_call__

        def experimental_monkey_patch_call(self, *args, **kwds):
            if (
                len(args) > 0
                and len(kwds) == 0
                and all((isinstance(x, FMGraphNode) for x in args))
            ):
                node = args[0]
                return node.apply_layer(self, *args[1:])
            else:
                local_value = module
                return __old_call__(self, *args, **kwds)

        module.__call__ = experimental_monkey_patch_call

    replace_call(module)


def disable_experimental_api():
    for module in predefined_modules:
        module.__call__ = module.__old_call__
