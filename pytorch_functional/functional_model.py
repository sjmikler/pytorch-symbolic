#  Copyright (c) 2022 Szymon Mikler

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Set, Tuple

import torch
from torch import nn

from . import configs
from .graph_algorithms import figure_out_nodes_between, topological_sort
from .symbolic import SymbolicTensor


class FunctionalModel(nn.Module):
    def __init__(
        self,
        inputs: Tuple[SymbolicTensor, ...] | List[SymbolicTensor] | SymbolicTensor,
        outputs: Tuple[SymbolicTensor, ...] | List[SymbolicTensor] | SymbolicTensor,
        enable_cuda_graphs=False,
    ):
        """A PyTorch model that applies operations defined by the graph.

        All operations that changed ``inputs`` into ``outputs`` will be applied
        in the same order on the real data that will be fed into this model.

        Parameters
        ----------
        inputs
            FMGraphNode object or a tuple of them.
        outputs
            FMGraphNode object or a tuple of them.
        """
        super().__init__()
        logging.info("Creating a Functional Model...")

        if isinstance(inputs, SymbolicTensor):
            inputs = (inputs,)
        assert all(isinstance(x, SymbolicTensor) for x in inputs)
        self.inputs: Tuple[SymbolicTensor, ...] = tuple(inputs)

        if isinstance(outputs, SymbolicTensor):
            outputs = (outputs,)
        assert all(isinstance(x, SymbolicTensor) for x in outputs)
        self.outputs: Tuple[SymbolicTensor, ...] = tuple(outputs)

        self._has_single_input = len(self.inputs) == 1
        self._has_single_output = len(self.outputs) == 1

        self._figure_out_execution_order()

        self._registered_modules: List[nn.Module] = []
        self._register_used_modules()

        self.cuda_graphs_enabled = False

        if enable_cuda_graphs:
            self._enable_cuda_graphs(self.inputs)

        if configs.MODULE_CALL_OPTIMIZATION:
            configs.remove_call_wrapper_from_all_modules()

    def forward(self, *inputs: torch.Tensor) -> Any:
        assert len(inputs) == len(self.inputs), "Number of inputs doesn't match!"
        for input_data, input_node in zip(inputs, self.inputs):
            input_node._launch_input(input_data)

        for node in self._execution_order:
            node._launch()

        if self._has_single_output:
            return self.outputs[0]._output
        else:
            return tuple(output_leaf._output for output_leaf in self.outputs)

    @property
    def input_shape(self):
        if self._has_single_input:
            return self.inputs[0].shape
        else:
            return tuple(node.shape for node in self.inputs)

    @property
    def output_shape(self):
        if self._has_single_output:
            return self.outputs[0].shape
        else:
            return tuple(node.shape for node in self.outputs)

    def _enable_cuda_graphs(self, inputs: Tuple[SymbolicTensor, ...]):
        msg = (
            "CUDA Graphs can result in undefined behaviour! "
            "Please read https://pytorch.org/docs/stable/notes/cuda.html#constraints."
        )
        logging.warning(msg)
        assert torch.cuda.is_available(), "CUDA acceleration is not available!"
        for x in inputs:
            assert x.batch_size_known, "Must provide batch size for each input!"

        self.cuda()
        input_tensors = tuple(x.v.cuda() for x in inputs)
        torch.cuda.make_graphed_callables(self, sample_args=input_tensors)
        self.cuda_graphs_enabled = True

    def _register_module(self, node: SymbolicTensor) -> bool:
        if not isinstance(node.layer, nn.Module):
            logging.info(f"Not registering {node.layer} (not a nn.Module)!")
            return False
        elif node.layer in self._registered_modules:
            logging.info(f"Not registering {node.layer} (already registered)!")
            return False

        num_modules = len(self._registered_modules)
        self.add_module(name=f"module{num_modules:0>3}_depth{node.depth:0>3}", module=node.layer)
        self._registered_modules.append(node.layer)
        return True

    @property
    def _used_nodes(self) -> Set[SymbolicTensor]:
        """Return a set of all nodes used in this model."""
        return figure_out_nodes_between(self.inputs, self.outputs)

    def _register_used_modules(self):
        num_registered = 0
        for node in self._execution_order:
            if self._register_module(node):
                num_registered += 1
        logging.info(f"Registered {num_registered} modules!")

    def _figure_out_execution_order(self):
        # Exclude inputs, as we don't launch any layers in input nodes
        used_nodes_excl_inputs = self._used_nodes - set(self.inputs)
        self._execution_order = topological_sort(used_nodes_excl_inputs)
