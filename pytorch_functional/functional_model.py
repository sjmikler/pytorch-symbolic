#  Copyright (c) 2022 Szymon Mikler

from __future__ import annotations

import logging
from typing import List, Tuple

import torch
from torch import nn

from . import configs
from .placeholders import Placeholder


class FunctionalModel(nn.Module):
    def __init__(self, inputs, outputs, enable_cuda_graphs=False):
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

        if isinstance(inputs, Placeholder):
            inputs = (inputs,)
        assert all(isinstance(x, Placeholder) for x in inputs)
        self.inputs = inputs

        if isinstance(outputs, Placeholder):
            outputs = (outputs,)
        assert all(isinstance(x, Placeholder) for x in outputs)
        self.outputs = outputs

        self._has_single_input = len(self.inputs) == 1
        self._has_single_output = len(self.outputs) == 1
        self._registered_modules = []
        self._prune_unused_layers()
        self._register_reachable_modules()

        self.cuda_graphs_enabled = False

        if enable_cuda_graphs:
            self._enable_cuda_graphs(inputs)

        if configs.MODULE_CALL_OPTIMIZATION:
            configs.remove_call_wrapper_from_all_modules()

    def forward(
        self, inputs: Tuple[torch.Tensor, ...] | torch.Tensor
    ) -> Tuple[torch.Tensor, ...] | torch.Tensor:
        if self._has_single_input:
            self.inputs[0]._begin_graph_flow(inputs)
        else:
            assert len(inputs) == len(self.inputs), "Number of inputs doesn't match!"
            for root, arg in zip(self.inputs, inputs):
                root._begin_graph_flow(arg)

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

    def _enable_cuda_graphs(self, inputs: Tuple[Placeholder]):
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

    def _register_module(self, node: Placeholder) -> bool:
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

    def _register_reachable_modules(self):
        reachable_nodes = self._reachable_nodes
        num_registered = 0
        for node in reachable_nodes:
            if self._register_module(node):
                num_registered += 1
        logging.info(f"Registered {num_registered} modules!")

    def _prune_unused_layers(self):
        used_nodes = self._used_nodes
        used_nodes_ids = {id(node) for node in used_nodes}
        reachable_nodes_ids = {id(node) for node in self._reachable_nodes}
        to_prune = reachable_nodes_ids.difference(used_nodes_ids)

        logging.info(f"Pruning {len(to_prune)} modules!")

        already_called = []
        for root in self.inputs:
            root._remove_outsiders_below(insiders=used_nodes, already_called=already_called)

    def _clear_nodes_memory(self):
        for node in self._reachable_nodes:
            node._clear_memory()

    @property
    def _reachable_nodes(self) -> List[Placeholder]:
        nodes_below: List[Placeholder] = []
        for root in self.inputs:
            root._get_all_nodes_below(nodes_below)
        return nodes_below

    @property
    def _used_nodes(self) -> List[Placeholder]:
        nodes_above: List[Placeholder] = []
        for output_leaf in self.outputs:
            output_leaf._get_all_nodes_above(nodes_above)
        return nodes_above
