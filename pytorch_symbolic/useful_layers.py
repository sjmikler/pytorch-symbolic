#  Copyright (c) 2022 Szymon Mikler

from typing import Callable

import torch
from torch import nn


class AnyOpLayer(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.forward = op


class NamedAnyOpLayer(nn.Module):
    def __init__(self, op, name):
        super().__init__()
        self.forward = op
        self.name = name

    def __repr__(self):
        return self.name

    def _get_name(self):
        return self.name


class CallbackLayer(nn.Module):
    def __init__(self, op):
        """This can be used for debugging or logging purposes. Accepts only one argument.

        Example::

            x = CallbackLayer(print)(x)

        It does not change anything in x, but prints its value.
        """
        super().__init__()
        self.callback = op

    def forward(self, arg):
        self.callback(arg)
        return arg


class AddOpLayer(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(a, b):
        return a + b


class SubOpLayer(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(a, b):
        return a - b


class MulOpLayer(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(a, b):
        return a * b


class ModOpLayer(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(a, b):
        return torch.remainder(a, b)


class MatmulOpLayer(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(a, b):
        return a @ b


class ConcatLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, *tensors):
        return torch.cat(tensors=tensors, dim=self.dim)


class StackLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, *tensors):
        return torch.stack(tensors=tensors, dim=self.dim)


class ReshapeLayer(nn.Module):
    def __init__(self, shape, batch_size_included=False):
        super().__init__()
        if not batch_size_included:
            self.shape = (-1, *shape)
        else:
            self.shape = shape

    def forward(self, tensor):
        return torch.reshape(input=tensor, shape=self.shape)


class ViewCopyLayer(nn.Module):
    def __init__(self, shape, batch_size_included=False):
        super().__init__()
        if not batch_size_included:
            self.shape = (-1, *shape)
        else:
            self.shape = shape

    def forward(self, tensor):
        return torch.view_copy(input=tensor, size=self.shape)


class AggregateLayer(nn.Module):
    def __init__(self, op: Callable, dim=None, keepdim=False):
        super().__init__()
        self.op = op

        self.dim = dim
        self.keepdim = keepdim
        if self.dim is None:
            setattr(self, "forward", self.forward_nodim)

    def forward(self, tensor):
        return self.op(input=tensor, dim=self.dim, keepdim=self.keepdim)

    def forward_nodim(self, tensor):
        return self.op(input=tensor)


class UnpackLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return args


class SliceLayer(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, arg):
        return arg[self.idx]
