#  Copyright (c) 2022 Szymon Mikler

from torch import nn

from .. import FunctionalModel, Input, layers, tools


def test_1():
    inputs = Input(shape=(3, 32, 32))
    x = inputs(nn.Conv2d(inputs.channels, 32, 3))(nn.ReLU())
    x = x(nn.Conv2d(x.channels, 64, 3))(nn.ReLU())
    block_1_output = x(nn.MaxPool2d(3))

    x = block_1_output(nn.Conv2d(block_1_output.channels, 64, 3, padding=1))(nn.ReLU())
    x = x(nn.Conv2d(x.channels, 64, 3, padding=1))(nn.ReLU())
    block_2_output = x + block_1_output

    x = block_2_output(nn.Conv2d(block_2_output.channels, 64, 3, padding=1))(nn.ReLU())
    x = x(nn.Conv2d(x.channels, 64, 3, padding=1))(nn.ReLU())
    block_3_output = x + block_2_output

    x = block_3_output(nn.Conv2d(x.channels, 64, 3))(nn.ReLU())
    x = x(nn.AvgPool2d(kernel_size=(x.H, x.W)))(nn.Flatten())
    x = x(nn.Linear(x.features, 256))(nn.ReLU())
    x = x(nn.Dropout(0.5))
    outputs = x(nn.Linear(x.features, 10))

    model = FunctionalModel(inputs, outputs)
    assert tools.get_parameter_count(model) == 223242


def test_2():
    inputs = Input((3, 128, 128))
    x = inputs

    x = x(nn.Conv2d(in_channels=x.channels, out_channels=16, kernel_size=3))
    x = x(nn.MaxPool2d(kernel_size=2))
    x = x(nn.ReLU())

    x = x(nn.Conv2d(in_channels=x.channels, out_channels=32, kernel_size=3))
    x = x(nn.MaxPool2d(kernel_size=2))
    x = x(nn.ReLU())

    x = x(nn.Conv2d(in_channels=x.channels, out_channels=64, kernel_size=3))
    x = x(nn.MaxPool2d(kernel_size=2))
    x = x(nn.ReLU())

    x = x(nn.Conv2d(in_channels=x.channels, out_channels=64, kernel_size=3))
    x = x(nn.MaxPool2d(kernel_size=2))
    x = x(nn.ReLU())

    x = x(nn.Flatten())
    outputs = x(nn.Linear(in_features=x.features, out_features=10))
    model = FunctionalModel(inputs=inputs, outputs=outputs)
    assert model.output_shape == (None, 10)
    assert tools.get_parameter_count(model) == 83562


def test_3():
    task1_input = Input(shape=(1, 28, 28))
    task2_input = Input(shape=(3, 32, 32))

    x = task1_input
    x = x(nn.Conv2d(x.channels, 16, 3))
    x = x(nn.MaxPool2d(3))(nn.ReLU())
    x = x(nn.Flatten())
    head1_out = x(nn.Linear(x.features, 200))

    x = task2_input
    x = x(nn.Conv2d(x.channels, 16, 3))
    x = x(nn.MaxPool2d(3))(nn.ReLU())
    x = x(nn.Flatten())
    head2_out = x(nn.Linear(x.features, 200))

    x = head1_out + head2_out
    x = x(nn.Linear(x.features, 400))(nn.ReLU())
    task1_outputs = x(nn.Linear(x.features, 10))
    task2_outputs = x(nn.Linear(x.features, 10))

    model = FunctionalModel(inputs=(task1_input, task2_input), outputs=(task1_outputs, task2_outputs))
    assert tools.get_parameter_count(model) == 614228


def test_4():
    x1 = Input(shape=(1, 2, 3))
    x2 = Input(shape=(5, 2, 3))
    x = x1(layers.ConcatLayer(dim=1), x2)
    assert x.shape == (None, 6, 2, 3)
