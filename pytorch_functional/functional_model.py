#  Copyright (c) 2022 Szymon Mikler

from __future__ import annotations

import copy
import logging
from types import MethodType
from typing import Any, Dict, List, Set, Tuple

import torch
from torch import nn

from . import code_generator, config
from .graph_algorithms import figure_out_nodes_between, topological_sort
from .symbolic_tensor import SymbolicTensor


class DetachedFunctionalModel(nn.Module):
    def __init__(self, names: List[str], layers: List[nn.Module], forward_src: str):
        """A tiny model detached from the FunctionalModel graph structure.

        It can live, even if the graph structure is removed!
        """
        super().__init__()
        self._execution_order_layers = []
        for name, layer in zip(names, layers):
            try:
                layer = copy.deepcopy(layer)
            except Exception as e:
                logging.error(f"Deepcopy of {layer} failed!")
                raise e
            self.add_module(name, layer)
            self._execution_order_layers.append(layer)

        self._generated_forward_source = forward_src

        scope = {"self": self}
        exec(self._generated_forward_source, {}, scope)
        self.forward = MethodType(scope["_generated_forward"], self)  # type: ignore


class FunctionalModel(nn.Module):
    def __init__(
        self,
        inputs: Tuple[SymbolicTensor, ...] | List[SymbolicTensor] | SymbolicTensor,
        outputs: Tuple[SymbolicTensor, ...] | List[SymbolicTensor] | SymbolicTensor,
        enable_cuda_graphs=False,
        enable_forward_codegen=True,
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

        # Initialize helper variables
        self._node_to_layer_name: Dict[SymbolicTensor, str] = {}
        self._layer_name_to_node: Dict[str, SymbolicTensor] = {}
        self._execution_order_nodes: List[SymbolicTensor] = []
        self._execution_order_layers: List[nn.Module] = []
        self._figure_out_execution_order()

        self._enable_forward_codegen = enable_forward_codegen
        if self._enable_forward_codegen:
            self._replace_forward_with_codegen()

        self._enable_cuda_graphs = enable_cuda_graphs
        if self._enable_cuda_graphs:
            self._convert_to_cuda_graphs(self.inputs)

    def forward(self, *inputs: torch.Tensor) -> Any:
        """This function is executed by __call__. Do not use this directly, use __call__ instead.

        Warning!

        This function will be overwritten by `_generate_optimized_forward` if `enable_forward_codegen`
        is True. If this happened and you want to see your source, print `self._generated_forward_source`.
        """
        assert len(inputs) == len(self.inputs), "Number of inputs doesn't match!"
        for input_data, input_node in zip(inputs, self.inputs):
            input_node._launch_input(input_data)

        for node in self._execution_order_nodes:
            node._launch()

        if len(self.outputs) == 1:
            return self.outputs[0]._output
        else:
            return tuple(output_leaf._output for output_leaf in self.outputs)

    @property
    def input_shape(self):
        """Return shape of the input or in case of multiple inputs - a tuple of them."""
        if len(self.inputs) == 1:
            return self.inputs[0].shape
        else:
            return tuple(node.shape for node in self.inputs)

    @property
    def output_shape(self):
        """Return shape of the output or in case of multiple outputs - a atuple of them."""
        if len(self.outputs):
            return self.outputs[0].shape
        else:
            return tuple(node.shape for node in self.outputs)

    def add_output(self, node: SymbolicTensor):
        assert node not in self.inputs, "Node is an input!"
        assert node in self._execution_order_nodes, "Node is not in the graph!"

        self.outputs = (*self.outputs, node)
        if self._enable_forward_codegen:
            self._replace_forward_with_codegen()

    def detach_from_graph(self) -> DetachedFunctionalModel:
        if self._enable_cuda_graphs:
            logging.warning("This might fail after converting to CUDA Graphs!")

        names = [self._node_to_layer_name[node] for node in self._execution_order_nodes]
        forward_src = code_generator.generate_forward_with_loops(
            self.inputs,
            self.outputs,
            self._execution_order_nodes,
            min_loop_length=config.CODEGEN_MIN_LOOP_LENGTH,
        )
        return DetachedFunctionalModel(names, self._execution_order_layers, forward_src)

    def _replace_forward_with_codegen(self):
        self._generated_forward_source = code_generator.generate_forward_with_loops(
            self.inputs,
            self.outputs,
            self._execution_order_nodes,
            min_loop_length=config.CODEGEN_MIN_LOOP_LENGTH,
        )
        scope = {"self": self}
        exec(self._generated_forward_source, {}, scope)
        self.forward = MethodType(scope["_generated_forward"], self)

    def _convert_to_cuda_graphs(self, inputs: Tuple[SymbolicTensor, ...]):
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

    @property
    def _used_nodes(self) -> Set[SymbolicTensor]:
        """Return a set of all nodes used in this model."""
        return figure_out_nodes_between(self.inputs, self.outputs)

    def _figure_out_execution_order(self):
        # Exclude inputs, as we don't launch any layers in input nodes
        used_nodes_excl_inputs = self._used_nodes - set(self.inputs)
        self._execution_order_nodes = topological_sort(used_nodes_excl_inputs)
        self._execution_order_layers = [node.layer for node in self._execution_order_nodes]

        num_layers = len(self._execution_order_nodes)
        str_length = len(str(num_layers))
        for idx, node in enumerate(self._execution_order_nodes):
            name = f"module{str(idx).zfill(str_length)}_depth{str(node.depth).zfill(str_length)}"
            self._layer_name_to_node[name] = node
            self._node_to_layer_name[node] = name
            self.add_module(name=name, module=node.layer)

    def __deepcopy__(self, memo):
        """This copies a working Module, but the underlying graph structure is not copied!"""
        obj = self.detach_from_graph()
        memo[id(self)] = obj
        return obj
