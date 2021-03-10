import torch
from torch import nn


class AddOpLayer(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(a, b):
        return a + b


class AnyOpLayer(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.forward = op


class ConcatOpLayer(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim - 1

    def forward(self, a, b):
        return torch.cat((a, b), dim=self.dim)


class StackOpLayer(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim - 1

    def forward(self, a, b):
        return torch.stack((a, b), dim=self.dim)

