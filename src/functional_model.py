import logging

import torch
from torch import nn

from . import layers


class FMGraphNode:
    def __init__(self, value: torch.Tensor, parents=tuple(), depth=0, layer=None):
        self._v = value
        self.parents = parents
        self.children = []
        self.layer = layer
        self.depth = depth
        self._output = None
        self._parents_outputs = []

    @property
    def channels(self):
        return self._v.shape[1]

    @property
    def features(self):
        return self._v.shape[1]

    @property
    def H(self):
        assert len(self._v.shape) == 4, "Variable is not an image!"
        return self._v.shape[2]

    @property
    def W(self):
        assert len(self._v.shape) == 4, "Variable is not an image!"
        return self._v.shape[3]

    @property
    def shape(self):
        return (None, *self._v.shape[1:])

    @property
    def numel(self):
        return self._v.shape[1:].numel()

    def apply_layer(self, layer, *others):
        if others:
            new_depth = min(self.depth, *(o.depth for o in others)) + 1
        else:
            new_depth = self.depth + 1

        new_output = layer.forward(self._v, *(o._v for o in others))

        new_layer_node = FMGraphNode(
            value=new_output, parents=(self, *others), layer=layer, depth=new_depth
        )
        self.children.append(new_layer_node)
        for other in others:
            other.children.append(new_layer_node)
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
            self._output = self.layer.forward(*self._parents_outputs)
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
        if isinstance(other, FMGraphNode):
            assert self.shape == other.shape, "Shapes do not match for the operation!"
            return self.apply_layer(layers.AddOpLayer(), other)
        else:
            return self.apply_layer(layers.AnyOpLayer(op=lambda x: x + other))

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, FMGraphNode):
            assert self.shape == other.shape, "Shapes do not match for the operation!"
            return self.apply_layer(layers.MulOpLayer(), other)
        else:
            return self.apply_layer(layers.AnyOpLayer(op=lambda x: x - other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self.apply_layer(layers.AnyOpLayer(op=lambda x: -x))

    def __pow__(self, other):
        if isinstance(other, FMGraphNode):
            assert self.shape == other.shape, "Shapes do not match for the operation!"
            return self.apply_layer(layers.AnyOpLayer(op=lambda x, y: x ** y), other)
        else:
            return self.apply_layer(layers.AnyOpLayer(op=lambda x: x ** other))

    def __rpow__(self, other):
        if isinstance(other, FMGraphNode):
            return other.__pow__(self)
        else:
            return self.apply_layer(layers.AnyOpLayer(op=lambda x: other ** x))

    def __sub__(self, other):
        if isinstance(other, FMGraphNode):
            assert self.shape == other.shape, "Shapes do not match for the operation!"
            return self.apply_layer(layers.SubOpLayer(), other)
        else:
            return self.apply_layer(layers.AnyOpLayer(op=lambda x: x - other))

    def __rsub__(self, other):
        if isinstance(other, FMGraphNode):
            return other.__sub__(self)
        else:
            return self.apply_layer(layers.AnyOpLayer(op=lambda x: other - x))

    def __truediv__(self, other):
        if isinstance(other, FMGraphNode):
            assert self.shape == other.shape, "Shapes do not match for the operation!"
            return self.apply_layer(layers.AnyOpLayer(op=lambda x, y: x / y), other)
        else:
            return self.apply_layer(layers.AnyOpLayer(op=lambda x: x / other))

    def __rtruediv__(self, other):
        if isinstance(other, FMGraphNode):
            return other.__truediv__(self)
        else:
            return self.apply_layer(layers.AnyOpLayer(op=lambda x: other / x))

    def __matmul__(self, other):
        if isinstance(other, FMGraphNode):
            return self.apply_layer(layers.MatmulOpLayer(), other)
        else:
            return self.apply_layer(layers.AnyOpLayer(op=lambda x: x @ other))

    def __rmatmul__(self, other):
        if isinstance(other, FMGraphNode):
            return other.__matmul__(self)
        else:
            return self.apply_layer(layers.AnyOpLayer(op=lambda x: other @ x))


class Input(FMGraphNode):
    def __init__(
        self,
        shape,
        dtype=torch.float32,
        min_value=0.0,
        max_value=1.0,
        _batch_size=1,
        _use_tensor=None,
    ):
        if _use_tensor is not None:
            super().__init__(value=_use_tensor)
            return

        value = torch.rand(_batch_size, *shape) * (max_value - min_value) + min_value
        value = value.to(dtype)
        super().__init__(value=value)


class FunctionalModel(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()

        if isinstance(inputs, FMGraphNode):
            inputs = (inputs,)
        assert all(isinstance(x, FMGraphNode) for x in inputs)
        self.inputs = inputs

        if isinstance(outputs, FMGraphNode):
            outputs = (outputs,)
        assert all(isinstance(x, FMGraphNode) for x in outputs)
        self.outputs = outputs

        self._has_single_input = len(self.inputs) == 1
        self._has_single_output = len(self.outputs) == 1
        self._registered_modules = []
        self._prune_unused_layers()
        self._register_reachable_modules()

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

    def _register_module(self, node):
        if (
                not isinstance(node.layer, nn.Module)
                or node.layer in self._registered_modules
        ):
            return False

        num_modules = len(self._registered_modules)
        self.add_module(
            name=f"module{num_modules:0>3}_depth{node.depth:0>3}", module=node.layer
        )
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
            root._remove_outsiders_below(insiders=used_nodes,
                                         already_called=already_called)

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
