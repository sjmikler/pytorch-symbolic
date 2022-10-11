# Quick Start

## Functional API for model creation

Deep neural networks can be represented as directed acyclic graphs where nodes are inputs, outputs or intermediate
states and edges are layers (in general transformations or functions). In such graph,
there exists a nonempty set of input nodes and a nonempty set of output nodes.
If your architecture meets the above conditions, it can be created in a functional manner.

### Introduction

To create a linear classificator without hidden layers, you can write following:

```python
from torch import nn
from pytorch_functional import Input, FunctionalModel

inputs = Input(shape=(28 * 28,))
outputs = nn.Linear(in_features=inputs.features, out_features=10)(inputs)
model = FunctionalModel(inputs, outputs)
```

Or equivalent:

```python
from torch import nn
from pytorch_functional import Input, FunctionalModel

inputs = Input(shape=(28 * 28,))
outputs = inputs(nn.Linear(in_features=inputs.features, out_features=10))
model = FunctionalModel(inputs, outputs)
```

To register a new layer, e.g. ``nn.Linear`` in Pytorch Functional, you can have two equivalent options:

* `layer(placeholder)` (like in Keras Functional API)
* `placeholder(layer)` (legacy)

Instead of deriving input and outputs shapes by hand, placeholder data is passed through the network.
Using its attributes we can instantly obtain shapes of intermediate outputs.

Doing this, we:

* Write less code
* Write easier code

### Comparing Pytorch Functional to Keras Functional

We took an example of a toy ResNet from [tensorflow guide](https://www.tensorflow.org/guide/keras/functional) and
created it in a few different ways. Note that their example is **16 lines long**, excluding imports and utilities.

Using Pytorch Functional, you can create toy ResNet using exactly as many lines as using Keras Functional:

```python
from torch import nn
from pytorch_functional import Input, FunctionalModel

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
```

In fact, each line of code in Pytorch Functional and Keras is functionally equivalent. For example this line in Keras:

```python
... = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
```

is equivalent to this line in Pytorch Functional:

```python
... = nn.Conv2d(x.channels, 64, 3, padding=1)(x)(nn.ReLU())
```

Let's analyze what happens in this line:

* `nn.Conv2d` is PyTorch equivalent of Keras `layers.Conv2d` layer
* Input channels:
    * In Keras we don't pass it openly - it'll be calculated automatically from the inputs
    * In Pytorch Functional we also calculate it automatically using `x.channels`, but we pass it openly as an argument
* In both frameworks `64, 3` are the output's number of channels and the kernel size
* Padding:
    * We use `padding="same"` in Keras
    * We use `padding=1` in PyTorch
* Activation:
    * In Keras we simply add argument `activation='relu'`
    * in Pytorch Functional we apply `nn.ReLU()` as a transformation that should happen after `nn.Conv2d(...)`

The example below is equivalent, but uses another way of registering layers in the network:

```python
from torch import nn
from pytorch_functional import Input, FunctionalModel

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
```

This took 16 lines of code.

You can register new layers in whichever way is more convinient for you.

### Contrast to vanilla PyTorch

A usual way to define the model in PyTorch is to create a class that inherits from `nn.Module`.

Steps:

1. define a class that inherits from `nn.Module`
2. define all the necessary layers in `__init__` method
    * You might have to calculate some things by hand: e.g. the number of input features for nn.Linear
3. define the order in which layers are executed in `forward` method

The separation of steps 2 and 3 often makes network creation tedious and more complicated than it should be.

PyTorch non-functional example (toy ResNet equivalent):

```python
from torch import nn


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
```

This took over 30 lines of code.

### [More functional API examples](https://github.com/gahaalt/pytorch-functional/tree/main/examples)

The link above includes:

* simple Encoder-Decoder architecture (coupling multiple FunctionalModels)
* VGG
* ResNet

## How to use (legacy way)

The section below explains how to create networks by using the `placeholder(layer)` method.

There are no differences between `placeholder(layer)` and `layer(placeholder)`. They both return the resulting placeholder as an output.

### Creating a functional model for RGB images step by step

1. Get placeholder input `inputs = Input(shape)`, where `shape` is in `(C, H, W)` format without batch dimension
    * If you need to provide the batch size as well, you should use `inputs = Input(batch_shape=(B, C, H, W)`.
2. To define the model, you can use placeholder's property `.shape`, or use its handy shortcuts:
    * `.features` for number of features in case of 1-dimensional data
    * `.C` or `.channels` for the number of channels
    * `.H` for the height of the image
    * `.W` for the width of the image
    * Placeholders have standard operations defined: `+`, `-`, `*`, `/`, `**` and `abs`.
      For example, `x = 2 + inputs` or `x = inputs / y` will work as expected.
    * Concatenate or stack placeholders using `pytorch_functional.layers.ConcatOpLayer` and `pytorch_functional.layers.StackOpLayer`
3. Register new modules in the model using: `outputs = inputs(layer)`
4. When all the layers are added, create the model: `model = FunctionalModel(inputs, outputs)`
5. Use `model` as a normal PyTorch model. It's 100% compatibile.

### Sequential topology example

```python
from torch import nn
from pytorch_functional import Input, FunctionalModel

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
```

### Multiple inputs example

```python
from torch import nn
from pytorch_functional import Input, FunctionalModel

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
```

## Special cases

### Use custom functions on Placeholders

If you want to use custom function or function from `torch.nn.functional` on placeholder symbolic variable,
you have to wrap it in a `nn.Module`.

For example:

```python
from torch import nn


def custom_func(*args, **kwds):
    ...


class CustomModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwds):
        return custom_func(*args)
```

There are more examples available in in the code of `pytorch_functional.layers`.

### Layer taking multiple arguments

You can do it in a moore natural way using `layer(placeholder)` notation:

```python
from pytorch_functional import Input, layers

x1 = Input(shape=(1, 2, 3))
x2 = Input(shape=(5, 2, 3))
x = layers.ConcatLayer(dim=1)(x1, x2)
x.shape  # (6, 2, 3)
```

Alternatively, using the other notation, do it like this `placeholder(layer, *other_placeholders)`:

```python
from pytorch_functional import Input, layers

x1 = Input(shape=(1, 2, 3))
x2 = Input(shape=(5, 2, 3))
x = x1(layers.ConcatLayer(dim=1), x2)
x.shape  # (6, 2, 3)
```

## Features

* Keras-like API
* Multiple inputs and outputs
* Automatic pruning of unused layers
* Reusing layers multiple times

## Limitations

* You cannot create a graph (model) with cycles

* You cannot use all custom functions on Placeholders:
    * this won't work: `x = Input((1, 2, 3)); x = torch.abs(x);`
    * but this will work: `x = Input((1, 2, 3)); x = abs(x);`
    * if needed, you can always create a `nn.Module` wrapper for `torch.abs`

## References

* [https://www.tensorflow.org/guide/keras/functional](https://www.tensorflow.org/guide/keras/functional)
