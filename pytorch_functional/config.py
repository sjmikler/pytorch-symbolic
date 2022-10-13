#  Copyright (c) 2022 Szymon Mikler

import logging

from .experimental_api import enable_functional_api_for_new_modules, remove_call_wrapper_from_all_modules

enable_functional_api_for_new_modules()


def optimize_module_calls():
    msg = "Optimizing module calls! Reusing modules might throw errors!"
    logging.warning(msg)
    remove_call_wrapper_from_all_modules()
