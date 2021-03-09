Defining models in tensorflow is easy: https://www.tensorflow.org/guide/keras/functional

I make it easy in PyTorch as well.

# Functional API for model creation

Deep learning models can be often presented as directed acyclic graphs with intermediate outputs as nodes and layers 
(aka. transformations, functions) as edges. In this graph, there exists a nonempty set of **input nodes**, which in fact are
nodes without any predecessors. Also, there exists a nonempty set of **output nodes**, which are nodes without any successors.

If your neural network meets the above conditions, it can be created in a functional manner.

# Model definition in PyTorch

An usual way to define a model in PyTorch is an objective one. Steps:

1. define a class that inherits from `nn.Module`
2. define all the layers, including shapes, in `__init__` method
3. define an order in which layers are used in `forward` method

The separation of step 2 and 3 makes network creation more difficult than it should be. Why?

* We have to know the exact shape of the input for each layer
* In more complicated networks, we have to create the model virtually twice: in `__init__` and in `forward`

Example (ResNet definition):

```
import torch
from torch import nn
from torch.nn import functional as F

class SimpleResBlock(nn.Module):
    def __init__(
            self,
            input_shape,
            stride,
            channels,
    ):
        super().__init__()
        x = torch.randn(*input_shape)

        self.conv0 = nn.Conv2d(
            in_channels=x.shape[1],
            out_channels=channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=(1, 1),
            bias=False,
        )
        x = self.conv0.forward(x)

        self.bn0 = nn.BatchNorm2d(num_features=x.shape[1])
        x = self.bn0.forward(x)

        self.conv1 = nn.Conv2d(
            in_channels=x.shape[1],
            out_channels=channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )
        x = self.conv1.forward(x)

    def forward(self, x):
        flow = x
        flow = self.conv0(flow)
        flow = self.bn0(flow)
        flow = F.relu(flow)
        flow = self.conv1(flow)
        return flow


class ResNet(nn.Module):
    def __init__(
            self,
            input_shape,
            n_classes,
            strides=(1, 2, 2),
            group_sizes=(2, 2, 2),
            features=(16, 32, 64),
    ):
        super().__init__()
        x = torch.randn(*input_shape)
        self.head = nn.Conv2d(in_channels=x.shape[1], out_channels=16,
                              kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        x = self.head.forward(x)

        self.shortcut_convs = []
        self.blocks = []
        for stride, group_size, channels in zip(strides, group_sizes, features):
            for _ in range(group_size):
                block = SimpleResBlock(
                    stride=stride, channels=channels, input_shape=x.shape)

                self.add_module(name=f'block{len(self.blocks)}', module=block)
                self.blocks.append(block)

                if stride != 1 or x.shape[1] != channels:
                    shortcut_conv = nn.Conv2d(
                        in_channels=x.shape[1],
                        out_channels=channels,
                        kernel_size=(1, 1),
                        stride=stride,
                        bias=False
                    )
                    self.shortcut_convs.append(shortcut_conv)
                else:
                    shortcut_conv = nn.Identity()
                    self.shortcut_convs.append(shortcut_conv)
                self.add_module(
                    name=f"shortcut{len(self.shortcut_convs)}", module=shortcut_conv)

                x = block.forward(x)
                stride = 1
        self.flatten = nn.Flatten()
        x = self.flatten.forward(x)
        self.classifier = nn.Linear(in_features=x.shape[1],
                                    out_features=n_classes)

    def forward(self, x):
        flow = self.head.forward(x)
        for block, shortcut_func in zip(self.blocks, self.shortcut_convs):
            outs = block.forward(flow)
            flow = shortcut_func(flow) + outs
        flow = self.flatten(flow)
        flow = self.classifier(flow)
        return flow

resnet = ResNet(input_shape=(1, 3, 32, 32), n_classes=10)
```

# Advantages of Functional API

In functional API, we create the neural network more naturally, as we would create a graph. Instead of defining layers
just to later decide how to connect intermediate states, we do it all at once. For example, after creating an input node
and a layer, we can instantly tell what shape will be the output of that layer and use this shape for creating next
layers.

Doing this we:

* Write less code
* Write easier code

Functional API example (ResNet definition):

```
from torch import nn
from pycharm.functional.FunctionalModel import FunctionalModel

def create_resnet(
        input_shape,
        n_classes,
        strides=(1, 2, 2),
        group_sizes=(2, 2, 2),
        features=(16, 32, 64),
):
    model = FunctionalModel(input_shape=input_shape)
    x = model.get_input()

    def create_block(x,
                     stride,
                     channels):
        x = x(nn.Conv2d(
            x.channels,
            out_channels=channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=(1, 1),
            bias=False
        ))
        x = x(nn.BatchNorm2d(num_features=x.features))
        x = x(nn.ReLU())
        x = x(nn.Conv2d(
            x.channels,
            out_channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False))
        return x

    x = x(nn.Conv2d(
        x.channels,
        out_channels=16,
        kernel_size=(3, 3),
        padding=(1, 1)))
    x = x(nn.ReLU())

    for stride, group_size, channels in zip(strides, group_sizes, features):
        for _ in range(group_size):
            x_main = create_block(x, stride, channels)

            if stride != 1 or channels != x.channels:
                x_short = x(nn.Conv2d(
                    x.channels,
                    out_channels=channels,
                    kernel_size=(1, 1),
                    stride=stride,
                    bias=False))
            else:
                x_short = x
            x = x_main + x_short
            stride = 1
    x = x(nn.Flatten())
    x = x(nn.Linear(x.features, out_features=n_classes))
    model.add_output(x, assert_shape=[n_classes])
    return model

resnet = create_resnet((3, 32, 32), n_classes=10)
```

# Features

- [x] Multiple outputs
- [ ] Multiple inputs
- [x] Pruning of unused layers
- [x] Reusing layers
- [x] Complex topologies
- [ ] Built-in graph plotting
- [ ] Stability (yes, it is a feature)
- [ ] Non-deterministic graphs

# References

* https://www.tensorflow.org/guide/keras/functional
