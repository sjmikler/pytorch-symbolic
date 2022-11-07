#  Copyright (c) 2022 Szymon Mikler

import logging
import os

from .experimental_api import enable_symbolic_API_for_new_modules, remove_call_wrapper_from_all_modules

enable_symbolic_API_for_new_modules()


def read_from_env(name, default):
    if name in os.environ:
        value = eval(os.environ[name], {}, {})
    else:
        value = default
    return value


# Constants
CODEGEN_BY_DEFAULT = read_from_env("PYTORCH_SYMBOLIC_CODEGEN_BY_DEFAULT", True)
CODEGEN_MIN_LOOP_LENGTH = read_from_env("PYTORCH_SYMBOLIC_CODEGEN_MIN_LOOP_LENGTH", 50)


def optimize_module_calls():
    """Remove the call wrapper from all existing `torch.nn.Module`."""
    msg = "Optimizing module calls for existing layers! Reusing them might throw errors!"
    logging.warning(msg)
    remove_call_wrapper_from_all_modules()
