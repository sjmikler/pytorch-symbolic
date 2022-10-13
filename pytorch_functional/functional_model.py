#  Copyright (c) 2022 Szymon Mikler

from __future__ import annotations

import logging
from types import MethodType
from typing import Any, Dict, List, Set, Tuple

import torch
from torch import nn

from .graph_algorithms import figure_out_nodes_between, topological_sort
from .symbolic_tensor import SymbolicTensor


class FunctionalModel(nn.Module):
    def __init__(
        self,
        inputs: Tuple[SymbolicTensor, ...] | List[SymbolicTensor] | SymbolicTensor,
        outputs: Tuple[SymbolicTensor, ...] | List[SymbolicTensor] | SymbolicTensor,
        enable_cuda_graphs=False,
        generate_optimized_forward=True,
    ):
        """A PyTorch model that applies operations defined by the graph.

        All operations that changed ``inputs`` into ``outputs`` will be applied
        in the same order on the real data that will be fed into this model.

        Example::

            input1 = Input((10,))
            input2 = Input((10,))
            x = input1 + input2
            x = nn.Linear(x.features, 1)(x)
            model = FunctionalModel((input1, input2), x)

        Parameters
        ----------
        inputs
            A collection of SymbolicTensors that will begin the computations.
            For these nodes, you'll provide the input. So if you have mulitple inputs,
            be prepared to pass multiple inputs during training/inference.

        outputs
            A collection of SymbolicTensors that will end the computations.
            These nodes return your final computation result.
            So if you have mulitple outputs, FunctionalModel will return a tuple of tensors.

        enable_cuda_graphs
            If True, after the model creation, model will be converted to CUDA Graph.
            This requires CUDA capable device.
            CUDA Graphs are greatly speeding up the execution of some of the models.
            Not all models are compatible with CUDA Graphs. For example, if your model
            includes some non-deterministic behaviour, it likely won't work.
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

        self._node_to_layer_name: Dict[SymbolicTensor, str] = {}
        self._layer_name_to_node: Dict[str, SymbolicTensor] = {}
        self._figure_out_execution_order()
        self._register_used_modules()

        self._generated_forward_source = None

        if generate_optimized_forward:
            self._generate_optimized_forward()

        self.cuda_graphs_enabled = False

        if enable_cuda_graphs:
            self._enable_cuda_graphs(self.inputs)

    def forward(self, *inputs: torch.Tensor) -> Any:
        """This function is executed by __call__. Do not use this directly, use __call__ instead.

        Warning!

        This function will be overwritten by `_generate_optimized_forward` if `generate_optimized_forward`
        is True. If this happened and you want to see your source, print `self._generated_forward_source`.
        """
        assert len(inputs) == len(self.inputs), "Number of inputs doesn't match!"
        for input_data, input_node in zip(inputs, self.inputs):
            input_node._launch_input(input_data)

        for node in self._execution_order_nodes:
            node._launch()

        if self._has_single_output:
            return self.outputs[0]._output
        else:
            return tuple(output_leaf._output for output_leaf in self.outputs)

    def _generate_optimized_forward(self):
        input_names = [node._get_str_name() for node in self.inputs]
        forward_definition = "def _generated_forward(self," + ",".join(input_names) + "):"
        code_lines = [forward_definition]

        TAB = " " * 4
        code_lines.append(TAB + "l=self._execution_order_layers")

        for exec_id, node in enumerate(self._execution_order_nodes):
            input_names = [node._get_str_name() for node in node.parents]
            output_name = node._get_str_name()
            code_line = TAB + output_name + f" = l[{exec_id}](" + ",".join(input_names) + ")"
            code_lines.append(code_line)

        return_line = TAB + "return " + ",".join(node._get_str_name() for node in self.outputs)
        code_lines.append(return_line)
        generated_forward = "\n".join(code_lines) + "\n"
        self._generated_forward_source = generated_forward

        exec(generated_forward, {}, locals())
        self.forward = MethodType(locals()["_generated_forward"], self)

    @property
    def input_shape(self):
        """Return shape of the input or in case of multiple inputs - a tuple of them."""
        if self._has_single_input:
            return self.inputs[0].shape
        else:
            return tuple(node.shape for node in self.inputs)

    @property
    def output_shape(self):
        """Return shape of the output or in case of multiple outputs - a atuple of them."""
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

    def _decide_name_for_node(self, node):
        assert node not in self._layer_name_to_node, "Node is already named!"
        name = f"module{len(self._layer_name_to_node)}_depth{node.depth:0>3}"
        self._layer_name_to_node[name] = node
        self._node_to_layer_name[node] = name
        return name

    @property
    def _used_nodes(self) -> Set[SymbolicTensor]:
        """Return a set of all nodes used in this model."""
        return figure_out_nodes_between(self.inputs, self.outputs)

    def _register_used_modules(self):
        for node in self._execution_order_nodes:
            assert isinstance(node.layer, nn.Module)
            self.add_module(name=self._node_to_layer_name[node], module=node.layer)

    def _figure_out_execution_order(self):
        # Exclude inputs, as we don't launch any layers in input nodes
        used_nodes_excl_inputs = self._used_nodes - set(self.inputs)
        self._execution_order_nodes = topological_sort(used_nodes_excl_inputs)
        self._execution_order_layers = []

        for node in self._execution_order_nodes:
            self._decide_name_for_node(node)
            self._execution_order_layers.append(node.layer)
