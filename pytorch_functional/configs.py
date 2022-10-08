#  Copyright (c) 2022 Szymon Mikler

import logging

from .experimental_api_v2 import enable_functional_api_for_new_modules, remove_call_wrapper_from_all_modules

enable_functional_api_for_new_modules()

DISABLED = 0
ENABLED = 1

MODULE_CALL_OPTIMIZATION = DISABLED


def enable_module_call_optimization():
    global MODULE_CALL_OPTIMIZATION
    msg = "Enabling call optimization! Reusing modules might throw errors!"
    logging.warning(msg)
    MODULE_CALL_OPTIMIZATION = ENABLED
    remove_call_wrapper_from_all_modules()
