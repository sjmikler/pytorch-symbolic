#  Copyright (c) 2022 Szymon Mikler

from torch import nn

from pytorch_functional import FunctionalModel, Input, model_tools


def test_on_resnet():
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

    model = FunctionalModel(inputs, outputs)
    bare_model = model.bare()

    assert model_tools.model_similar(model, bare_model)
    assert model_tools.models_have_corresponding_parameters(model, bare_model)
