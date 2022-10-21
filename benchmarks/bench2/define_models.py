#  Copyright (c) 2022 Szymon Mikler

from torch import nn

from pytorch_symbolic import Input, SymbolicModel


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

        # does not work with varying image size
        # kernel_size = 7  # calculated by hand
        # self.global_pool = nn.AvgPool2d(kernel_size)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

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
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.relu(self.linear(x))
        x = self.dropout(x)
        return self.classifier(x)


def functional_toy_resnet(bs, cuda_graphs, img_size):
    inputs = Input(batch_shape=(bs, 3, img_size, img_size))

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
    x = x(nn.AdaptiveAvgPool2d((1, 1)))(nn.Flatten())
    x = x(nn.Linear(x.features, 256))(nn.ReLU())
    x = x(nn.Dropout(0.5))
    outputs = x(nn.Linear(x.features, 10))

    return SymbolicModel(inputs, outputs, enable_cuda_graphs=cuda_graphs)


def create_toy_resnets(bs, img_size):
    """ToyResNet example from https://www.tensorflow.org/guide/keras/functional#a_toy_resnet_model."""
    models = [
        (("functional",), functional_toy_resnet(bs, cuda_graphs=False, img_size=img_size)),
        (("functional", "cuda_graphs"), functional_toy_resnet(bs, cuda_graphs=True, img_size=img_size)),
        (("vanilla",), ToyResNet()),
    ]
    return models
