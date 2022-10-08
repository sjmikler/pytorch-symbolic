"""
This is a flexible implementation of VGG architecture.
"""

#  Copyright (c) 2022 Szymon Mikler

from torch import nn

from .. import FunctionalModel, Input
from .common import classifier


def VGG(
    input_shape,
    n_classes,
    version=None,
    group_sizes=(1, 1, 2, 2, 2),
    channels=(64, 128, 256, 512, 512),
    pools=(2, 2, 2, 2, 2),
    activation=nn.ReLU(),
    final_pooling="avgpool",
    **kwargs,
):
    if kwargs:
        print(f"VGG: unknown parameters: {kwargs.keys()}")
    if version:
        if version == 11:
            group_sizes = (1, 1, 2, 2, 2)
        elif version == 13:
            group_sizes = (2, 2, 2, 2, 2)
        elif version == 16:
            group_sizes = (2, 2, 3, 3, 3)
        elif version == 19:
            group_sizes = (2, 2, 4, 4, 4)
        else:
            raise NotImplementedError(f"Unkown version={version}!")

    inputs = Input(shape=input_shape)
    flow = inputs

    iteration = 0
    for group_size, width, pool in zip(group_sizes, channels, pools):
        if iteration == 0:
            iteration = 1
        else:
            flow = flow(nn.MaxPool2d(pool))

        for _ in range(group_size):
            flow = flow(nn.Conv2d(flow.channels, width, 3, 1, 1, bias=False))
            flow = flow(nn.BatchNorm2d(flow.channels))(activation)

    outs = classifier(flow, n_classes, final_pooling)
    model = FunctionalModel(inputs=inputs, outputs=outs)
    return model
