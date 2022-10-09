#  Copyright (c) 2022 Szymon Mikler

import torch

import examples
from pytorch_functional import tools


def test_toy_resnet():
    model = examples.resnet.ToyResNet((3, 32, 32), n_classes=10)
    inputs = torch.rand(16, 3, 32, 32)
    outputs = model(inputs)
    assert list(outputs.shape) == [16, 10]
    assert tools.get_parameter_count(model) == 175530


def test_wrn():
    model = examples.resnet.ResNet((3, 32, 32), n_classes=10, version=("WRN", 16, 4))
    inputs = torch.rand(16, 3, 32, 32)
    outputs = model(inputs)
    assert list(outputs.shape) == [16, 10]
    assert tools.get_parameter_count(model) == 2750698


def test_vgg():
    model = examples.vgg.VGG((3, 32, 32), n_classes=10, version=13)
    inputs = torch.rand(16, 3, 32, 32)
    outputs = model(inputs)
    assert list(outputs.shape) == [16, 10]
    assert tools.get_parameter_count(model) == 9413066


def test_enc_dec():
    model = examples.encoder_decoder.simple_encoder_decoder((3, 32, 32))
    inputs = torch.rand(16, 3, 32, 32)
    outputs = model(inputs)
    assert list(outputs.shape) == [16, 1, 7, 7]
    assert tools.get_parameter_count(model) == 28529
