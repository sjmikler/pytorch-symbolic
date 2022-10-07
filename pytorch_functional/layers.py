#  Copyright (c) 2022 Szymon Mikler

import torch
from torch import nn


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


class MatmulOpLayer(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(a, b):
        return a @ b


class AnyOpLayer(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.forward = op


class ConcatLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, *elements):
        return torch.cat(tensors=elements, dim=self.dim)


class StackLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, *elements):
        return torch.stack(tensors=elements, dim=self.dim)


class ReshapeLayer(nn.Module):
    def __init__(self, shape, batch_size_included=False):
        super().__init__()
        if not batch_size_included:
            self.shape = (-1, *shape)
        else:
            self.shape = shape

    def forward(self, inputs):
        return torch.reshape(input=inputs, shape=self.shape)
