import torch
from torch import nn
from pytorch_functional import Input, FunctionalModel


def test_toyresnet():
    class ToyResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.block1conv1 = nn.Conv2d(3, 32, 3)
            self.block1conv2 = nn.Conv2d(32, 64, 3)
            self.maxpool = nn.MaxPool2d(3)

            self.block2conv1 = nn.Conv2d(64, 64, 3, padding=1)
            self.block2conv2 = nn.Conv2d(64, 64, 3, padding=1)

            self.block3conv1 = nn.Conv2d(64, 64, 3, padding=1)
            self.block3conv2 = nn.Conv2d(64, 64, 3, padding=1)

            self.conv1 = nn.Conv2d(64, 64, 3)

            kernel_size = 7  # calculated by hand
            self.global_pool = nn.AvgPool2d(kernel_size)
            self.flatten = nn.Flatten()
            self.linear = nn.Linear(64, 256)
            self.dropout = nn.Dropout(0.5)
            self.classifier = nn.Linear(256, 10)

        def forward(self, x):
            x = self.relu(self.block1conv1(x))
            x = self.relu(self.block1conv2(x))
            block_1_output = self.maxpool(x)

            x = self.relu(self.block2conv1(block_1_output))
            x = self.relu(self.block2conv2(x))
            block_2_output = x + block_1_output

            x = self.relu(self.block3conv1(block_2_output))
            x = self.relu(self.block3conv2(x))
            block_3_output = x + block_2_output

            x = self.relu(self.conv1(block_3_output))
            x = self.global_pool(x)
            x = self.flatten(x)
            x = self.relu(self.linear(x))
            x = self.dropout(x)
            return self.classifier(x)

    model = ToyResNet()
    inputs = torch.rand(1, 3, 32, 32)
    outs = model.forward(inputs)
    assert tuple(outs.shape) == (1, 10)


def test_func_resnet():
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
    inputs = torch.rand(1, 3, 32, 32)
    outs = model.forward(inputs)
    assert tuple(outs.shape) == (1, 10)


def test_simple_linear():
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
    inputs = torch.rand(1, 3, 128, 128)
    outs = model.forward(inputs)
    assert tuple(outs.shape) == (1, 10)


def test_multiple_input():
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

    model = FunctionalModel(inputs=(task1_input, task2_input),
                            outputs=(task1_outputs, task2_outputs))
    outs = model.forward((torch.rand(1, 1, 28, 28), torch.rand(1, 3, 32, 32)))
    assert len(outs) == 2
    assert tuple(outs[0].shape) == (1, 10)
    assert tuple(outs[1].shape) == (1, 10)


def test_func_to_layer():
    x = Input(shape=(5, 6, 7))
    x = x(torch.abs)
