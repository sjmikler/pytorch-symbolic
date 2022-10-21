#  Copyright (c) 2022 Szymon Mikler

import logging

from .experimental_api import enable_functional_api_for_new_modules, remove_call_wrapper_from_all_modules

enable_functional_api_for_new_modules()

# Constants
CODEGEN_MIN_LOOP_LENGTH = 50


def optimize_module_calls():
    msg = "Optimizing module calls for existing layers! Reusing them might throw " "errors!"
    logging.warning(msg)
    remove_call_wrapper_from_all_modules()
