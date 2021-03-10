import torch
from torch import nn
from pytorch_functional.layers import SummingLayer


class FunctionalModel(nn.Module):
    def __init__(self, input_shape, _batch_size=1):
        super().__init__()
        self.input_shape = input_shape
        self.input_value = torch.randn(_batch_size, *input_shape)
        self.output_leaves = []
        self._modules_added = []
        self.root = FMGraphNode(self.input_value,
                                functional_model=self)

    def get_input(self):
        return self.root

    def add_output(self, output, assert_shape=None):
        assert isinstance(output, FMGraphNode)

        if assert_shape:
            if isinstance(assert_shape, int):
                assert_shape = [assert_shape]
            assert all(
                x == y for x, y in zip(list(output.shape)[1:], assert_shape)
            )

        if output not in self.output_leaves:
            self.output_leaves.append(output)

    def forward(self, *args):
        self.root._forward_edge(*args)
        assert len(self.output_leaves) > 0

        if len(self.output_leaves) == 1:
            return self.output_leaves[0].outs
        else:
            return (output_leaf.outs for output_leaf in self.output_leaves)

    def _register_module(self, node):
        num_modules = len(self._modules_added)
        self.add_module(name=f"module{num_modules:0>3}_depth{node.depth:0>3}",
                        module=node.layer)
        self._modules_added.append(node.layer)

    def _prune_unused_layers(self):
        nodes_above = []
        for output_leaf in self.output_leaves:
            output_leaf._get_all_nodes_above(nodes_above)

        self.root._remove_outsiders_below(insiders=nodes_above)

    @property
    def _reachable_nodes(self):
        nodes_below = []
        self.root._get_all_nodes_below(nodes_below)
        return nodes_below


class FMGraphNode:
    def __init__(
            self,
            value: torch.Tensor,
            functional_model: FunctionalModel,
            parents=[],
            depth=0,
            layer=None
    ):
        self.v = value
        self.functional_model = functional_model
        self.parents = parents
        self.children = []
        self.layer = layer
        self.real_inputs = []
        self.depth = depth

    @property
    def channels(self):
        return self.v.shape[1]

    @property
    def features(self):
        return self.v.shape[1]

    @property
    def in_channels(self):
        return self.v.shape[1]

    @property
    def in_features(self):
        return self.v.shape[1]

    @property
    def num_features(self):
        return self.v.shape[1]

    @property
    def shape(self):
        return self.v.shape

    def apply_layer(self, layer):
        new_output = layer.forward(self.v)
        new_layer_node = FMGraphNode(
            value=new_output,
            functional_model=self.functional_model,
            parents=[self],
            layer=layer,
            depth=self.depth + 1
        )
        self.functional_model._register_module(new_layer_node)
        self.children.append(new_layer_node)
        return new_layer_node

    def _get_root(self):
        root = self
        while root.parents:
            root = root.parents[0]
        return root

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

    def __call__(self, layer):
        return self.apply_layer(layer)

    def __add__(self, other):
        assert self.shape == other.shape, "Shapes do not match for addition!"

        layer = SummingLayer()
        new_layer_node = FMGraphNode(
            value=layer.forward(self.v, other.v),
            functional_model=self.functional_model,
            parents=[self, other],
            layer=layer,
            depth=self.depth + 1,
        )
        self.functional_model._register_module(new_layer_node)
        self.children.append(new_layer_node)
        other.children.append(new_layer_node)
        return new_layer_node

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
