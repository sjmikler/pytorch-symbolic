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

