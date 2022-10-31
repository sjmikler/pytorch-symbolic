#  Copyright (c) 2022 Szymon Mikler

from __future__ import annotations

import logging
from typing import Any, List, Set, Tuple

import torch
from torch import nn

from . import useful_layers


class SymbolicData:
    def __init__(
        self,
        value: Any,
        parents: Tuple[SymbolicData, ...] = tuple(),
        depth: int = 0,
        layer: nn.Module | None = None,
        batch_size_known: bool = False,
    ):
        """Grandfather of all Symbolic datatypes."""
        self.v = value
        self.layer = layer
        self.depth = depth
        self.batch_size_known = batch_size_known

        self._output = None
        self._children: List[SymbolicData] = []
        self._parents: Tuple[SymbolicData, ...] = parents
        self._layer_full_siblings: Tuple[SymbolicData, ...] = (self,)

    @property
    def parents(self) -> Tuple[SymbolicData, ...]:
        """Acces the tuple of parents of this node."""
        return tuple(self._parents)

    @property
    def children(self) -> Tuple[SymbolicData, ...]:
        """Acces the tuple of children of this node."""
        return tuple(self._children)

    def __len__(self) -> int:
        """Length of the symbolic data."""
        return len(self.v)

    def apply_module(
        self, layer: nn.Module, *others: SymbolicData
    ) -> SymbolicData | Tuple[SymbolicData, ...]:
        """Register a new layer in the graph. Layer must be nn.Module."""
        assert all([isinstance(other, SymbolicData) for other in others]), "Works with SymbolicData only!"

        parents = (self, *others)
        new_depth = max(parent.depth for parent in parents) + 1
        new_output = layer.__call__(self.v, *(o.v for o in others))

        cls = SymbolicTensor if isinstance(new_output, torch.Tensor) else SymbolicData

        new_layer_node = cls(
            value=new_output,
            parents=parents,
            layer=layer,
            depth=new_depth,
            batch_size_known=self.batch_size_known,
        )
        for parent in parents:
            parent._children.append(new_layer_node)
            logging.info(f"Added {new_layer_node} as child of {parent}")
        return new_layer_node

    def __iter__(self):
        """Creates the only layer that has multiple children from one operation.

        Suitable for unpacking results, even nested ones.
        """
        layer = useful_layers.UnpackLayer()
        new_outputs = layer.__call__(*self.v)

        new_layer_nodes = []
        for new_output in new_outputs:

            cls = SymbolicData
            if isinstance(new_output, torch.Tensor):
                cls = SymbolicTensor

            new_layer_nodes.append(
                cls(
                    value=new_output,
                    parents=(self,),
                    layer=layer,
                    depth=self.depth + 1,
                    batch_size_known=self.batch_size_known,
                )
            )
        for new_layer_node in new_layer_nodes:
            new_layer_node._layer_full_siblings = tuple(new_layer_nodes)

        self._children.extend(new_layer_nodes)
        for new_layer_node in new_layer_nodes:
            logging.info(f"Added {new_layer_node} as child of {self}")
        for node in new_layer_nodes:
            yield node

    def _get_all_nodes_above(self) -> Set[SymbolicData]:
        nodes_seen = {self}
        to_expand = [self]
        while to_expand:
            node = to_expand.pop()
            for parent in node._parents:
                if parent not in nodes_seen:
                    to_expand.append(parent)
                    nodes_seen.add(parent)
        return nodes_seen

    def _get_all_nodes_below(self) -> Set[SymbolicData]:
        nodes_seen = {self}
        to_expand = [self]
        while to_expand:
            node = to_expand.pop()
            for child in node._children:
                if child not in nodes_seen:
                    to_expand.append(child)
                    nodes_seen.add(child)
        return nodes_seen

    def _launch_input(self, x):
        self._output = x

    def _launch(self):
        if len(self._layer_full_siblings) > 1:
            assert len(self._parents) == 1
            outputs = self.layer(*self._parents[0]._output)
            for node, out in zip(self._layer_full_siblings, outputs):
                node._output = out
        else:
            self._output = self.layer(*(parent._output for parent in self._parents))

    def __getitem__(self, idx):
        layer = useful_layers.SliceLayer(idx)
        return layer(self)

    def __call__(self, *args):
        return self.apply_module(*args)

    def __repr__(self):
        addr = f"SymbolicData({self.v.__class__.__name__.capitalize()}) at {hex(id(self))};"
        info = f"{len(self._parents)} parents; {len(self._children)} children"
        return "<" + addr + " " + info + ">"

    def __hash__(self):
        return id(self)


class SymbolicTensor(SymbolicData):
    def __init__(self, *args, **kwds):
        """Most common Symbolic datatype. It mimics ``torch.Tensor``."""
        super().__init__(*args, **kwds)
        assert isinstance(self.v, torch.Tensor)

    @property
    def features(self) -> int:
        """Size of 1D data."""
        assert len(self.v.shape) == 2, "The data is not of [C,F] form!"
        return self.v.shape[1]

    @property
    def C(self) -> int:
        """Number of channels in Image data."""
        assert len(self.v.shape) == 4, "The data is not of [C,H,W] form!"
        return self.v.shape[1]

    @property
    def channels(self) -> int:
        """Same as ``.C``"""
        return self.C

    @property
    def H(self) -> int:
        """Height in Image data."""
        assert len(self.v.shape) == 4, "The data is not of [C,H,W] form!"
        return self.v.shape[2]

    @property
    def W(self) -> int:
        """Width in Image data."""
        assert len(self.v.shape) == 4, "The data is not of [C,H,W] form!"
        return self.v.shape[3]

    @property
    def HW(self) -> Tuple[int, int]:
        """Tuple of (height, width) in Image data."""
        return (self.H, self.W)

    @property
    def CHW(self) -> Tuple[int, int, int]:
        """Tuple of (channels, height, width) in Image data."""
        return (self.C, self.H, self.W)

    @property
    def HWC(self) -> Tuple[int, int, int]:
        """Tuple of (height, width, channels) in Image data."""
        return (self.H, self.W, self.C)

    @property
    def batch_size(self) -> int | None:
        """Batch size of the data. Will be default if was not provided."""
        return self.v.shape[0]

    @property
    def shape(self) -> Tuple[int | None, ...]:
        """Shape of the placeholder, including batch size."""
        return self.v.shape

    @property
    def numel(self) -> int:
        """Number of the values in placeholder. If batch size is known, it is used too."""
        return self.v.shape.numel()

    def reshape(self, *shape) -> SymbolicTensor:
        reshape_layer = useful_layers.ReshapeLayer(batch_size_included=True, shape=shape)
        return reshape_layer(self)

    def view(self, *shape) -> SymbolicTensor:
        view_copy_layer = useful_layers.ViewCopyLayer(batch_size_included=True, shape=shape)
        return view_copy_layer(self)

    def t(self) -> SymbolicTensor:
        transpose_layer = useful_layers.AnyOpLayer(op=lambda x: x.t())
        return transpose_layer(self)

    @property
    def T(self) -> SymbolicTensor:
        return self.t()

    def mean(self, dim=None, keepdim=False) -> SymbolicTensor:
        layer = useful_layers.AggregateLayer(torch.mean, dim=dim, keepdim=keepdim)
        return layer(self)

    def sum(self, dim=None, keepdim=False) -> SymbolicTensor:
        layer = useful_layers.AggregateLayer(torch.sum, dim=dim, keepdim=keepdim)
        return layer(self)

    def median(self, dim=None, keepdim=False) -> SymbolicTensor:
        layer = useful_layers.AggregateLayer(torch.median, dim=dim, keepdim=keepdim)
        return layer(self)

    def argmax(self, dim=None, keepdim=False) -> SymbolicTensor:
        layer = useful_layers.AggregateLayer(torch.argmax, dim=dim, keepdim=keepdim)
        return layer(self)

    def argmin(self, dim=None, keepdim=False) -> SymbolicTensor:
        layer = useful_layers.AggregateLayer(torch.argmin, dim=dim, keepdim=keepdim)
        return layer(self)

    def flatten(self) -> SymbolicTensor:
        return nn.Flatten()(self)

    def __abs__(self):
        return self(useful_layers.AnyOpLayer(lambda x: abs(x)))

    def __add__(self, other):
        if isinstance(other, SymbolicTensor):
            return self(useful_layers.AddOpLayer(), other)
        else:
            return self(useful_layers.AnyOpLayer(op=lambda x: x + other))

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, SymbolicTensor):
            return self(useful_layers.MulOpLayer(), other)
        else:
            return self(useful_layers.AnyOpLayer(op=lambda x: x * other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mod__(self, other):
        if isinstance(other, SymbolicTensor):
            return self(useful_layers.ModOpLayer(), other)
        else:
            return self(useful_layers.AnyOpLayer(op=lambda x: x % other))

    def __rmod__(self, other):
        return self(useful_layers.AnyOpLayer(op=lambda x: other % x))

    def __neg__(self):
        return self(useful_layers.AnyOpLayer(op=lambda x: -x))

    def __pow__(self, other):
        if isinstance(other, SymbolicTensor):
            return self(useful_layers.AnyOpLayer(op=lambda x, y: x**y), other)
        else:
            return self(useful_layers.AnyOpLayer(op=lambda x: x**other))

    def __rpow__(self, other):
        return self(useful_layers.AnyOpLayer(op=lambda x: other**x))

    def __sub__(self, other):
        if isinstance(other, SymbolicTensor):
            return self(useful_layers.SubOpLayer(), other)
        else:
            return self(useful_layers.AnyOpLayer(op=lambda x: x - other))

    def __rsub__(self, other):
        return self(useful_layers.AnyOpLayer(op=lambda x: other - x))

    def __truediv__(self, other):
        if isinstance(other, SymbolicTensor):
            return self(useful_layers.AnyOpLayer(op=lambda x, y: x / y), other)
        else:
            return self(useful_layers.AnyOpLayer(op=lambda x: x / other))

    def __rtruediv__(self, other):
        return self(useful_layers.AnyOpLayer(op=lambda x: other / x))

    def __matmul__(self, other):
        if isinstance(other, SymbolicTensor):
            return self(useful_layers.MatmulOpLayer(), other)
        else:
            return self(useful_layers.AnyOpLayer(op=lambda x: x @ other))

    def __rmatmul__(self, other):
        return self(useful_layers.AnyOpLayer(op=lambda x: other @ x))

    def __repr__(self):
        addr = f"SymbolicTensor at {hex(id(self))};"
        info = f"{len(self._parents)} parents; {len(self._children)} children"
        return "<" + addr + " " + info + ">"


def Input(
    shape: Tuple | List = tuple(),
    batch_size: int = 1,
    batch_shape: Tuple | List | None = None,
    dtype=torch.float32,
    min_value: float = 0.0,
    max_value: float = 1.0,
):
    """Input to Symbolic Model. Creates Symbolic Tensor as a root node in the graph.

    It should be treated as a placeholder that will be replaced with real data after the model is created.
    For calculation purposes, it can be treated as normal ``torch.Tensor``,
    which means it can be added, subtracted, multiplied, taken absolute value of, etc.

    Parameters
    ----------
    shape
        Shape of the real data NOT including the batch dimension
    batch_size
        Optional batch size of the Tensor
    batch_shape
        Shape of the real data including the batch dimension.
        Should be provided instead ``shape`` if cuda graphs will be used.
        If both ``shape`` and ``batch_shape`` are given, ``batch_shape`` has higher priority.
    dtype
        Dtype of the real data that will be the input of the network
    min_value
        In rare cases, if real world data is very specific and some values
        cannot work with the model, this should be used to set a
        reasonable minimal value that the model can take as an input.
    max_value
        As above, but the maximal value
    """
    batch_size_known = True

    if batch_shape is not None:
        batch_size = batch_shape[0]
        shape = batch_shape[1:]
    else:
        # By default, we use batch_size of 1 under the hood
        batch_size_known = False

    value = torch.rand(batch_size, *shape) * (max_value - min_value) + min_value
    value = value.to(dtype)
    return SymbolicTensor(value=value, batch_size_known=batch_size_known)


def CustomInput(
    data: Any,
):
    """Input to Symbolic Model. Creates Symbolic Data as a root node in the graph.

    This should be used when Input won't work.

    Parameters
    ----------
    data
        Speficic data that will be used during the graph tracing.
        It can, but doesn't need to be a torch.Tensor.
    """
    if isinstance(data, torch.Tensor):
        return SymbolicTensor(value=data, batch_size_known=True)
    else:
        return SymbolicData(value=data)
