#  Copyright (c) 2022 Szymon Mikler

from torch import nn

from .. import layers


def classifier(flow, n_classes, pooling="avgpool"):
    if pooling == "catpool":
        maxp = flow(nn.MaxPool2d(kernel_size=(flow.H, flow.W)))
        avgp = flow(nn.AvgPool2d(kernel_size=(flow.H, flow.W)))
        flow = maxp(layers.ConcatLayer(dim=1), avgp)(nn.Flatten())
    if pooling == "avgpool":
        flow = flow(nn.AvgPool2d(kernel_size=(flow.H, flow.W)))(nn.Flatten())
    if pooling == "maxpool":
        flow = flow(nn.MaxPool2d(kernel_size=(flow.H, flow.W)))(nn.Flatten())
    return flow(nn.Linear(flow.features, n_classes))
