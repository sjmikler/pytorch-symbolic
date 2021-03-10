import torch
from torch import nn
from pytorch_functional.source import layers
import logging


class FMGraphNode:
    def __init__(self, value: torch.Tensor, parents=tuple(), depth=0, layer=None):
        self._v = value
        self.parents = parents
        self.children = []
        self.layer = layer
        self.depth = depth
        self.real_inputs = []

    @property
    def channels(self):
        return self._v.shape[1]

    @property
    def features(self):
        return self._v.shape[1]

    @property
    def H(self):
        assert len(self.shape) == 3, "Variable is not an image!"
        return self._v.shape[2]

    @property
    def W(self):
        assert len(self.shape) == 3, "Variable is not an image!"
        return self._v.shape[3]

    @property
    def shape(self):
        return tuple(self._v.shape[1:])

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

    def _remove_outsiders_below(self, insiders):
        for child in self.children.copy():
            if child in insiders:
                child._remove_outsiders_below(insiders)
            else:
                self.children.remove(child)

    def _forward_edge(self, x):
        if len(self.parents) == 0:
            for child in self.children:
                child._forward_edge(x)

        self.real_inputs.append(x)

        if len(self.real_inputs) == len(self.parents):
            self.outs = self.layer.forward(*self.real_inputs)
            self.real_inputs = []
            for child in self.children:
                child._forward_edge(self.outs)

    def __call__(self, *args):
        return self.apply_layer(*args)

    def __abs__(self):
        return self.apply_layer(layers.AnyOpLayer(lambda x: abs(x)))

    def __add__(self, other):
        assert self.shape == other.shape, "Shapes do not match for the operation!"
        return self.apply_layer(layers.AddOpLayer(), other)

    def __mul__(self, other):
        assert self.shape == other.shape, "Shapes do not match for the operation!"
        return self.apply_layer(layers.AnyOpLayer(op=lambda x, y: x * y), other)

    def __neg__(self):
        return self.apply_layer(layers.AnyOpLayer(lambda x: -x))

    def __pow__(self, other):
        assert self.shape == other.shape, "Shapes do not match for the operation!"
        return self.apply_layer(layers.AnyOpLayer(op=lambda x, y: x ** y), other)

    def __sub__(self, other):
        assert self.shape == other.shape, "Shapes do not match for the operation!"
        return self.apply_layer(layers.AnyOpLayer(op=lambda x, y: x - y), other)

    def __truediv__(self, other):
        assert self.shape == other.shape, "Shapes do not match for the operation!"
        return self.apply_layer(layers.AnyOpLayer(op=lambda x, y: x / y), other)


class Input(FMGraphNode):
    def __init__(self, shape, batch_size=1, dtype=torch.float32, min_value=0., max_value=1.):
        value = torch.rand(batch_size, *shape) * (max_value - min_value) + min_value
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

        self._registered_modules = []
        self._prune_unused_layers()
        self._register_reachable_modules()

    def forward(self, *args):
        if len(self.inputs) == 1:
            self.inputs[0]._forward_edge(*args)
        else:
            assert len(args) == 1, "Pass multiple inputs in a tuple!"
            assert len(args[0]) == len(self.inputs), "Numbers of inputs don't match!"
            for root, arg in zip(self.inputs, args[0]):
                if isinstance(arg, tuple):
                    root._forward_edge(*arg)
                else:
                    root._forward_edge(arg)

        if len(self.outputs) == 1:
            return self.outputs[0].outs
        else:
            return tuple(output_leaf.outs for output_leaf in self.outputs)

    def _register_module(self, node):
        if node.layer in self._registered_modules or node.layer is None:
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
        reachable_nodes = self._reachable_nodes
        logging.info(f"Pruning {len(reachable_nodes) - len(used_nodes)} modules!")
        for root in self.inputs:
            root._remove_outsiders_below(insiders=used_nodes)

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
