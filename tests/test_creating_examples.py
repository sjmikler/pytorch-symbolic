#  Copyright (c) 2022 Szymon Mikler

import torch
from torch import nn

import examples
from pytorch_symbolic import Input, SymbolicModel, model_tools


def test_example_toy_resnet():
    model = examples.resnet.ToyResNet((3, 32, 32), n_classes=10)
    inputs = torch.rand(16, 3, 32, 32)
    outputs = model(inputs)
    assert list(outputs.shape) == [16, 10]
    assert model_tools.get_parameter_count(model) == 175530


def test_example_wrn():
    model = examples.resnet.ResNet((3, 32, 32), n_classes=10, version=("WRN", 16, 4))
    inputs = torch.rand(16, 3, 32, 32)
    outputs = model(inputs)
    assert list(outputs.shape) == [16, 10]
    assert model_tools.get_parameter_count(model) == 2750698


def test_example_vgg():
    model = examples.vgg.VGG((3, 32, 32), n_classes=10, version=13)
    inputs = torch.rand(16, 3, 32, 32)
    outputs = model(inputs)
    assert list(outputs.shape) == [16, 10]
    assert model_tools.get_parameter_count(model) == 9413066


def test_example_enc_dec():
    model = examples.encoder_decoder.simple_encoder_decoder((3, 32, 32))
    inputs = torch.rand(16, 3, 32, 32)
    outputs = model(inputs)
    assert list(outputs.shape) == [16, 1, 7, 7]
    assert model_tools.get_parameter_count(model) == 28529


def test_resnet():
    inputs = Input(shape=(3, 32, 32))
    x = nn.Conv2d(inputs.channels, 32, 3)(inputs)(nn.ReLU())
    x = nn.Conv2d(x.channels, 64, 3)(x)(nn.ReLU())
    block_1_output = nn.MaxPool2d(3)(x)

    x = nn.Conv2d(block_1_output.channels, 64, 3, padding=1)(block_1_output)(nn.ReLU())
    x = nn.Conv2d(x.channels, 64, 3, padding=1)(x)(nn.ReLU())
    block_2_output = x + block_1_output

    x = nn.Conv2d(block_2_output.channels, 64, 3, padding=1)(block_2_output)(nn.ReLU())
    x = nn.Conv2d(x.channels, 64, 3, padding=1)(x)(nn.ReLU())
    block_3_output = x + block_2_output

    x = nn.Conv2d(x.channels, 64, 3)(block_3_output)(nn.ReLU())
    x = nn.AvgPool2d(kernel_size=(x.H, x.W))(x)(nn.Flatten())
    x = nn.Linear(x.features, 256)(x)(nn.ReLU())
    x = nn.Dropout(0.5)(x)
    outputs = nn.Linear(x.features, 10)(x)

    model = SymbolicModel(inputs, outputs)
    assert model_tools.get_parameter_count(model) == 223242


def test_lstm():
    examples.lstm.run()
