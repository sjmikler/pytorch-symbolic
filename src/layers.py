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

    def forward(self, *elements) :
        return torch.stack(tensors=elements, dim=self.dim)
