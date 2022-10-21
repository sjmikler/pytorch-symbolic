#  Copyright (c) 2022 Szymon Mikler

from __future__ import annotations

import logging
from typing import List, Set, Tuple

import torch
from torch import nn

from . import useful_layers


class SymbolicTensor:
    def __init__(
        self,
        value: torch.Tensor,
        parents: Tuple[SymbolicTensor, ...] = tuple(),
        depth: int = 0,
        layer: nn.Module | None = None,
        batch_size_known: bool = False,
    ):
        """Node in a Functional Model.

        This might represent input or intermediate values of the neural network.

        Parameters
        ----------
        value
            x object representing example of value that will flow through the node.
        parents
            Tuple of parent nodes - other Placeholders.
        depth
            How deep the node is in the tree. It is defined as maximum of parents' depth plus 1.
        layer
            nn.Module object that transformed parents into this object.
        batch_size_known
            If False, the batch size will be replaced with ``None`` when displaying it.
        """
        self.v = value
        self.parents = parents
        self.children: List[SymbolicTensor] = []
        self.layer = layer
        self.depth = depth
        self._output = None
        self._parents_outputs: List[torch.Tensor] = []
        self.batch_size_known = batch_size_known

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
        """If known - batch size of the data. Else None."""
        if self.batch_size_known:
            return self.v.shape[0]
        else:
            return None

    @property
    def shape(self) -> Tuple[int | None, ...]:
        """Shape of the placeholder, including batch size."""
        if self.batch_size_known:
            return self.v.shape
        else:
            return (None, *self.v.shape[1:])

    @property
    def numel(self) -> int:
        """Number of the values in placeholder. If batch size is known, it is used too."""
        return self.v.shape.numel()

    def apply_module(self, layer: nn.Module, *others: SymbolicTensor) -> SymbolicTensor:
        """Register a new layer in the graph. Layer must be nn.Module."""
        assert all([isinstance(other, SymbolicTensor) for other in others])

        parents = (self, *others)
        new_depth = max(parent.depth for parent in parents) + 1
        new_output = layer.__call__(self.v, *(o.v for o in others))
        batch_size_known = all([parent.batch_size_known for parent in parents])

        new_layer_node = SymbolicTensor(
            value=new_output,
            parents=parents,
            layer=layer,
            depth=new_depth,
            batch_size_known=batch_size_known,
        )
        for parent in parents:
            parent.children.append(new_layer_node)
            logging.info(f"Added {new_layer_node} as child of {parent}")
        return new_layer_node

    def _get_all_nodes_above(self) -> Set[SymbolicTensor]:
        nodes_seen = {self}
        to_expand = [self]
        while to_expand:
            node = to_expand.pop()
            for parent in node.parents:
                if parent not in nodes_seen:
                    to_expand.append(parent)
                    nodes_seen.add(parent)
        return nodes_seen

    def _get_all_nodes_below(self) -> Set[SymbolicTensor]:
        nodes_seen = {self}
        to_expand = [self]
        while to_expand:
            node = to_expand.pop()
            for child in node.children:
                if child not in nodes_seen:
                    to_expand.append(child)
                    nodes_seen.add(child)
        return nodes_seen

    def _launch_input(self, x):
        self._output = x

    def _launch(self):
        self._output = self.layer(*(parent._output for parent in self.parents))

    def __call__(self, *args):
        return self.apply_module(*args)

    def __abs__(self):
        return self(useful_layers.AnyOpLayer(lambda x: abs(x)))

    def __add__(self, other):
        if isinstance(other, SymbolicTensor):
            assert self.shape == other.shape, "Shapes do not match for the operation!"
            return self(useful_layers.AddOpLayer(), other)
        else:
            return self(useful_layers.AnyOpLayer(op=lambda x: x + other))

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, SymbolicTensor):
            assert self.shape == other.shape, "Shapes do not match for the operation!"
            return self(useful_layers.MulOpLayer(), other)
        else:
            return self(useful_layers.AnyOpLayer(op=lambda x: x * other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mod__(self, other):
        if isinstance(other, SymbolicTensor):
            assert self.shape == other.shape, "Shapes do not match for the operation!"
            return self(useful_layers.ModOpLayer(), other)
        else:
            return self(useful_layers.AnyOpLayer(op=lambda x: x % other))

    def __rmod__(self, other):
        if isinstance(other, SymbolicTensor):
            return other.__mod__(self)
        else:
            return self(useful_layers.AnyOpLayer(op=lambda x: other % x))

    def __neg__(self):
        return self(useful_layers.AnyOpLayer(op=lambda x: -x))

    def __pow__(self, other):
        if isinstance(other, SymbolicTensor):
            assert self.shape == other.shape, "Shapes do not match for the operation!"
            return self(useful_layers.AnyOpLayer(op=lambda x, y: x**y), other)
        else:
            return self(useful_layers.AnyOpLayer(op=lambda x: x**other))

    def __rpow__(self, other):
        if isinstance(other, SymbolicTensor):
            return other.__pow__(self)
        else:
            return self(useful_layers.AnyOpLayer(op=lambda x: other**x))

    def __sub__(self, other):
        if isinstance(other, SymbolicTensor):
            assert self.shape == other.shape, "Shapes do not match for the operation!"
            return self(useful_layers.SubOpLayer(), other)
        else:
            return self(useful_layers.AnyOpLayer(op=lambda x: x - other))

    def __rsub__(self, other):
        if isinstance(other, SymbolicTensor):
            return other.__sub__(self)
        else:
            return self(useful_layers.AnyOpLayer(op=lambda x: other - x))

    def __truediv__(self, other):
        if isinstance(other, SymbolicTensor):
            assert self.shape == other.shape, "Shapes do not match for the operation!"
            return self(useful_layers.AnyOpLayer(op=lambda x, y: x / y), other)
        else:
            return self(useful_layers.AnyOpLayer(op=lambda x: x / other))

    def __rtruediv__(self, other):
        if isinstance(other, SymbolicTensor):
            return other.__truediv__(self)
        else:
            return self(useful_layers.AnyOpLayer(op=lambda x: other / x))

    def __matmul__(self, other):
        if isinstance(other, SymbolicTensor):
            return self(useful_layers.MatmulOpLayer(), other)
        else:
            return self(useful_layers.AnyOpLayer(op=lambda x: x @ other))

    def __rmatmul__(self, other):
        if isinstance(other, SymbolicTensor):
            return other.__matmul__(self)
        else:
            return self(useful_layers.AnyOpLayer(op=lambda x: other @ x))

    def __repr__(self):
        addr = f"SymbolicTensor at {hex(id(self))};"
        info = f"child of {len(self.parents)}; parent of {len(self.children)}"
        return "<" + addr + " " + info + ">"

    def __hash__(self):
        return id(self)


class Input(SymbolicTensor):
    def __init__(
        self,
        shape: Tuple | List | None = None,
        include_batch: bool = True,
        batch_shape: Tuple | List | None = None,
        dtype=torch.float32,
        min_value: float = 0.0,
        max_value: float = 1.0,
        custom_tensor: torch.Tensor | None = None,
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
        include_batch
            If True and ``batch_shape`` was not given, batch_shape will not be added to ``shape``.
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
        custom_tensor
            If needed, a specific tensor can be provided to serve as the SymbolicTensor's value.
            If this is the case, no shape or dtype is needed as they will be inferred from the tensor.
        """
        if custom_tensor is not None:
            self.was_batch_size_provided = True
            super().__init__(value=custom_tensor, batch_size_known=True)
            return

        if batch_shape is None and not include_batch:
            batch_shape = shape

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
