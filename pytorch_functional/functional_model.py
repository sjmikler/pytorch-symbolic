#  Copyright (c) 2022 Szymon Mikler

import logging
from typing import Any, List

import torch
from torch import nn

from . import configs, layers


class Placeholder:
    def __init__(self, value: torch.Tensor, parents=tuple(), depth=0, layer=None, batch_size_known=False):
        """Node of a Functional Model.

        This might represent input or intermediate values of the neural network.

        Parameters
        ----------
        value
            Tensor object representing value of the node.
        parents
            Tuple of parent nodes.
        depth
            How deep the node is in the tree.
        layer
            nn.Module object that transformed parents.
        """
        self.v = value
        self.parents = parents
        self.children: List[Placeholder] = []
        self.layer = layer
        self.depth = depth
        self._output = None
        self._parents_outputs: List[Any] = []
        self.batch_size_known = batch_size_known

    @property
    def features(self):
        assert len(self.v.shape) == 2, "The data is not of [C,F] form!"
        return self.v.shape[1]

    @property
    def C(self):
        assert len(self.v.shape) == 4, "The data is not of [C,H,W] form!"
        return self.v.shape[1]

    @property
    def channels(self):
        return self.C

    @property
    def H(self):
        assert len(self.v.shape) == 4, "The data is not of [C,H,W] form!"
        return self.v.shape[2]

    @property
    def W(self):
        assert len(self.v.shape) == 4, "The data is not of [C,H,W] form!"
        return self.v.shape[3]

    @property
    def HW(self):
        return (self.H, self.W)

    @property
    def CHW(self):
        return (self.C, self.H, self.W)

    @property
    def HWC(self):
        return (self.H, self.W, self.C)

    @property
    def batch_size(self):
        if self.batch_size_known:
            return self.v.shape[0]
        else:
            return None

    @property
    def shape(self):
        if self.batch_size_known:
            return self.v.shape
        else:
            return (None, *self.v.shape[1:])

    @property
    def numel(self):
        return self.v.shape.numel()

    def apply_layer(self, layer, *others):
        all_parents = (self,) + others
        new_depth = min(parent.depth for parent in all_parents) + 1
        new_output = layer.__call__(self.v, *(o.v for o in others))
        batch_size_known = all([parent.batch_size_known for parent in all_parents])

        new_layer_node = Placeholder(
            value=new_output,
            parents=all_parents,
            layer=layer,
            depth=new_depth,
            batch_size_known=batch_size_known,
        )
        for parent in all_parents:
            parent.children.append(new_layer_node)
            logging.info(f"Added {new_layer_node} as child of {parent}")
        return new_layer_node

    def _get_all_nodes_below(self, layer_list):
        if self in layer_list:
            return layer_list
        layer_list.append(self)

        for child in self.children:
            child._get_all_nodes_below(layer_list)
        return layer_list

    def _get_all_nodes_above(self, layer_list):
        if self in layer_list:
            return layer_list
        layer_list.append(self)

        for child in self.parents:
            child._get_all_nodes_above(layer_list)
        return layer_list

    def _remove_outsiders_below(self, insiders, already_called):
        if self in already_called:
            return None
        already_called.append(self)

        for child in self.children.copy():
            if child in insiders:
                child._remove_outsiders_below(insiders, already_called)
            else:
                self.children.remove(child)

    def _forward_edge(self, x):
        if len(self.parents) == 0:
            for child in self.children:
                child._forward_edge(x)
            return

        self._parents_outputs.append(x)

        if len(self._parents_outputs) == len(self.parents):
            self._output = self.layer.__call__(*self._parents_outputs)
            self._parents_outputs = []
            for child in self.children:
                child._forward_edge(self._output)

    def _clear_memory(self):
        self._parents_outputs = []
        self._output = None

    def __call__(self, *args, **kwargs):
        return self.apply_layer(*args, **kwargs)

    def __abs__(self):
        return self.apply_layer(layers.AnyOpLayer(lambda x: abs(x)))

    def __add__(self, other):
        if isinstance(other, Placeholder):
            assert self.shape == other.shape, "Shapes do not match for the operation!"
            return self.apply_layer(layers.AddOpLayer(), other)
        else:
            return self.apply_layer(layers.AnyOpLayer(op=lambda x: x + other))

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, Placeholder):
            assert self.shape == other.shape, "Shapes do not match for the operation!"
            return self.apply_layer(layers.MulOpLayer(), other)
        else:
            return self.apply_layer(layers.AnyOpLayer(op=lambda x: x - other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self.apply_layer(layers.AnyOpLayer(op=lambda x: -x))

    def __pow__(self, other):
        if isinstance(other, Placeholder):
            assert self.shape == other.shape, "Shapes do not match for the operation!"
            return self.apply_layer(layers.AnyOpLayer(op=lambda x, y: x ** y), other)
        else:
            return self.apply_layer(layers.AnyOpLayer(op=lambda x: x ** other))

    def __rpow__(self, other):
        if isinstance(other, Placeholder):
            return other.__pow__(self)
        else:
            return self.apply_layer(layers.AnyOpLayer(op=lambda x: other ** x))

    def __sub__(self, other):
        if isinstance(other, Placeholder):
            assert self.shape == other.shape, "Shapes do not match for the operation!"
            return self.apply_layer(layers.SubOpLayer(), other)
        else:
            return self.apply_layer(layers.AnyOpLayer(op=lambda x: x - other))

    def __rsub__(self, other):
        if isinstance(other, Placeholder):
            return other.__sub__(self)
        else:
            return self.apply_layer(layers.AnyOpLayer(op=lambda x: other - x))

    def __truediv__(self, other):
        if isinstance(other, Placeholder):
            assert self.shape == other.shape, "Shapes do not match for the operation!"
            return self.apply_layer(layers.AnyOpLayer(op=lambda x, y: x / y), other)
        else:
            return self.apply_layer(layers.AnyOpLayer(op=lambda x: x / other))

    def __rtruediv__(self, other):
        if isinstance(other, Placeholder):
            return other.__truediv__(self)
        else:
            return self.apply_layer(layers.AnyOpLayer(op=lambda x: other / x))

    def __matmul__(self, other):
        if isinstance(other, Placeholder):
            return self.apply_layer(layers.MatmulOpLayer(), other)
        else:
            return self.apply_layer(layers.AnyOpLayer(op=lambda x: x @ other))

    def __rmatmul__(self, other):
        if isinstance(other, Placeholder):
            return other.__matmul__(self)
        else:
            return self.apply_layer(layers.AnyOpLayer(op=lambda x: other @ x))

    def __repr__(self):
        addr = f"Placeholder at {hex(id(self))};"
        info = f"child of {len(self.parents)}; parent of {len(self.children)}"
        return "<" + addr + " " + info + ">"


class Input(Placeholder):
    def __init__(
        self,
        shape=None,
        batch_shape=None,
        dtype=torch.float32,
        min_value=0.0,
        max_value=1.0,
        custom_tensor=None,
    ):
        """Input to the Functional Model.

        It should be treated as a placeholder value that will be replaced with
        real data after the model is created.
        For calculation purposes, it can be treated as a normal numerical object,
        which means it can be added, subtracted, multiplied, taken absolute value of,
        etc.

        Parameters
        ----------
        shape
            Shape of the real data NOT including the batch dimension.
        batch_shape
            Shape of the real data including the batch dimension.
            Should be provided instead ``shape`` if cuda graphs will be used.
            If both ``shape`` and ``batch_shape`` are given, ``batch_shape`` has higher priority.
        dtype
            Dtype of the real data that will be the input of the network.
        min_value
            In rare cases, if real world data is very specific and some values
            cannot work with the model, this should be used to set the
            reasonable minimal value that the model will work on.
        max_value
            As above, but the maximal value.
        """
        if custom_tensor is not None:
            self.was_batch_size_provided = True
            super().__init__(value=custom_tensor, batch_size_known=True)
            return

        if batch_shape is not None:
            batch_size = batch_shape[0]
            shape = batch_shape[1:]
            batch_size_known = True
        elif shape is not None:
            # We use batch_size of 1 under the hood
            # but we don't tell it to the user
            batch_size = 1
            batch_size_known = False
        else:
            raise ValueError("Shape argument is required!")

        value = torch.rand(batch_size, *shape) * (max_value - min_value) + min_value
        value = value.to(dtype)
        super().__init__(value=value, batch_size_known=batch_size_known)


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

    def forward(self, inputs):
        if self._has_single_input:
            self.inputs[0]._forward_edge(inputs)
        else:
            assert len(inputs) == len(self.inputs), "Number of inputs doesn't match!"
            for root, arg in zip(self.inputs, inputs):
                root._forward_edge(arg)

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

    def _enable_cuda_graphs(self, inputs):
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

    def _register_module(self, node):
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
    def _reachable_nodes(self):
        nodes_below = []
        for root in self.inputs:
            root._get_all_nodes_below(nodes_below)
        return nodes_below

    @property
    def _used_nodes(self):
        nodes_above = []
        for output_leaf in self.outputs:
            output_leaf._get_all_nodes_above(nodes_above)
        return nodes_above
