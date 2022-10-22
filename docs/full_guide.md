# Full guide

## Sections to do

* Introduction
* Training models
* Reusing graph for multiple models
* Reusing models in graphs
* Multiple in out
* Shared layers
* Extract nodes from symbolic model
* Add custom layers
* Mix and match styles


# Quick Start

## Symbolic API for model creation

Deep neural networks can be represented as directed acyclic graphs where:

* nodes are inputs, outputs or intermediate states
* edges are layers (in general transformations or functions)

In such graph,
there exists a nonempty set of input nodes and a nonempty set of output nodes.
If your architecture meets the above conditions, it can be created in a symbolic manner.

### Features

* Keras-like API
* Multiple inputs and outputs
* Layers can be shared between models
* Works with any user-defined module
* Produces fast models with CUDA Graphs acceleration available

## Introduction

To create a linear classificator without hidden layers, you can write the following:

```python
from torch import nn
from pytorch_symbolic import Input, SymbolicModel

inputs = Input(shape=(28 * 28,))
outputs = nn.Linear(in_features=inputs.features, out_features=10)(inputs)
model = SymbolicModel(inputs, outputs)
```

Or equivalently:

```python
from torch import nn
from pytorch_symbolic import Input, SymbolicModel

inputs = Input(shape=(28 * 28,))
outputs = inputs(nn.Linear(in_features=inputs.features, out_features=10))
model = SymbolicModel(inputs, outputs)
```

To register a new layer, e.g. ``nn.Linear`` in Pytorch Symbolic,
you can have two equivalent options:

* `layer(symbolic_tensor)` (like in Keras Symbolic API)
* `symbolic_tensor(layer)` (like nowhere elese)

There are no differences between the above.
They both create identical models and they both return a SymbolicTensor.

In Pytorch Symbolic symbolic data is passed through the network during its creation.
Using its attributes we can instantly obtain shapes of intermediate outputs,
instead of deriving them by hand. For example:

```py
print(inputs.shape)
print(inputs.features)
```

```
(None, 784)
784
```

Doing this, we:

* Write less code
* Write easier code

## Comparing Pytorch Symbolic to Keras Symbolic

We took an example of a toy ResNet from
[tensorflow guide](https://www.tensorflow.org/guide/keras/symbolic) and
recreated it in a few different ways. Note that their example is **16 lines long**,
excluding imports and utilities.

Using Pytorch Symbolic, you can create toy ResNet using exactly as many lines as using Keras:

```python
from torch import nn
from pytorch_symbolic import Input, SymbolicModel

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
x = nn.AvgPool2d(kernel_size=x.HW)(x)(nn.Flatten())
x = nn.Linear(x.features, 256)(x)(nn.ReLU())
x = nn.Dropout(0.5)(x)
outputs = nn.Linear(x.features, 10)(x)

model = SymbolicModel(inputs, outputs)
```

In fact, each line of code in Pytorch Symbolic and Keras is symbolcily equivalent. For example this line in Keras:

```python
... = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
```

is equivalent to this line in Pytorch Symbolic:

```python
... = nn.Conv2d(x.channels, 64, 3, padding=1)(x)(nn.ReLU())
```

Let's analyze what happens in this line:

* `nn.Conv2d` is PyTorch equivalent of Keras `layers.Conv2d` layer
* Input channels:
    * In Keras we don't pass it openly - it'll be calculated automatically from the inputs
    * In Pytorch Symbolic we also calculate it automatically using `x.channels`, but we pass it openly as an argument
* In both frameworks `64, 3` is the output number of channels and size of the kernel
* Padding:
    * We use `padding="same"` in Keras
    * We use `padding=1` in PyTorch
* Activation:
    * In Keras we simply add an argument `activation='relu'`
    * in Pytorch Symbolic we add `nn.ReLU()` as a transformation that happens after `nn.Conv2d(...)`

The example below is equivalent, but uses another way of registering layers in the network:

```python
from torch import nn
from pytorch_symbolic import Input, SymbolicModel

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

model = SymbolicModel(inputs, outputs)
```

This took 16 lines of code.

You can register new layers in whichever way you want or you can mix them.

### Comparison to vanilla PyTorch

A usual way to define the model in PyTorch is to create a class that inherits from `nn.Module`.

Steps:

1. define a class that inherits from `nn.Module`
2. define all the necessary layers in `__init__` method
    * You might have to calculate some things by hand: e.g. the number of input features for nn.Linear
3. define the order in which layers are executed in `forward` method

The separation of steps 2 and 3 often makes network creation tedious and more complicated than it should be.

PyTorch non-symbolic example (toy ResNet equivalent from previous section):

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

## [More symbolic API examples](https://github.com/gahaalt/pytorch-symbolic/tree/main/examples)

The link above includes:

* simple Encoder-Decoder architecture (coupling multiple SymbolicModels)
* VGG
* ResNet

### Creating a model for RGB images step by step

1. Get your symbolic input. These are ways to do it:
    * `inputs = Input(shape=(C, H, W))`
    * `inputs = Input(shape=(B, C, H, W), batched=False)`
    * `inputs = Input(batch_shape=(B, C, H, W))`
2. To define the model, you can use placeholder's property `.shape` or use handy shortcuts:
    * `.features` equals `.shape[1]` in case of 1-dimensional data
    * `.C` and `.channels` equals `.shape[1]` for the RGB data
    * `.H` equals `.shape[2]` for the RGB data
    * `.W` equals `.shape[3]` for the RGB data
    * Placeholders have standard operations defined: `+`, `-`, `*`, `/`, `**` and `abs`.
      For example, `x=2+inputs` or `x=inputs%y` will work as expected.
    * To concatenate (stacking is similar) symbolic layers:
        * use `useful_layers.ConcatOpLayer(dim=1)`
        * add custom function to the model:
          ```
          from pytorch_symbolic.functions_utility import add_to_model
          add_to_model(torch.concat, tensors=(...), dim=1)
          ```
3. Register new module in the model: `outputs = inputs(layer)` or `outputs = layer(inputs)`
4. When all the layers are added, create the model: `model = SymbolicModel(inputs, outputs)`
5. Use `model` as a normal PyTorch `nn.Module`. It's 100% compatibile.

### Sequential topology example

```python
from torch import nn
from pytorch_symbolic import Input, SymbolicModel

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
model = SymbolicModel(inputs=inputs, outputs=outputs)
assert model.output_shape == (None, 10)
```

### Multiple inputs example

```python
from torch import nn
from pytorch_symbolic import Input, SymbolicModel

task1_input = Input(shape=(1, 28, 28))
task2_input = Input(shape=(3, 32, 32))

x = task1_input
x = x(nn.Conv2d(x.channels, 16, 3))
x = x(nn.MaxPool2d(3))(nn.ReLU())(nn.Flatten())
head1_out = x(nn.Linear(x.features, 200))

x = task2_input
x = x(nn.Conv2d(x.channels, 16, 3))
x = x(nn.MaxPool2d(3))(nn.ReLU())(nn.Flatten())
head2_out = x(nn.Linear(x.features, 200))

x = head1_out + head2_out
x = x(nn.Linear(x.features, 400))(nn.ReLU())
task1_outputs = x(nn.Linear(x.features, 10))
task2_outputs = x(nn.Linear(x.features, 10))

model = SymbolicModel(inputs=(task1_input, task2_input), outputs=(task1_outputs, task2_outputs))
```

## Special cases

### Use custom functions on Placeholders

If you want to use custom function or function from `torch.nn.functional`
on placeholder symbolic variable, the recommended way is to wrap it in a `nn.Module`.

For example:

```python
from torch import nn


def custom_func(*args):
    print("Arguments: ", *args)
    return args


class CustomModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwds):
        return custom_func(*args)
```

There are more examples available in the code of `pytorch_symbolic.layers`.

### Layer taking multiple arguments

You can create layers with multiple inputs/outputs in a natural way using `layer(placeholder)` notation.

Just pass multiple arguments to the layer:

```python
from pytorch_symbolic import Input, useful_layers

x1 = Input(shape=(1, 2, 3))
x2 = Input(shape=(5, 2, 3))
x = useful_layers.ConcatLayer(dim=1)(x1, x2)
x.shape  # (6, 2, 3)
```

Alternatively, using the other notation, do it like this `placeholder(layer, *other_placeholders)`:

```python
from pytorch_symbolic import Input, useful_layers

x1 = Input(shape=(1, 2, 3))
x2 = Input(shape=(5, 2, 3))
x = x1(useful_layers.ConcatLayer(dim=1), x2)
x.shape  # (6, 2, 3)
```

### Adding custom functions to the model

If for some reason you want your model to execute a custom function, you can register it in the graph.

The recommended way is to convert functions into `nn.Module`,
but if you want, Pytorch Symbolic will do it for you:

```python
import torch
from pytorch_symbolic import Input
from pytorch_symbolic.functions_utility import add_to_model

x1 = Input(shape=(1, 2, 3))
x2 = Input(shape=(5, 2, 3))
x = add_to_model(torch.concat, (x1, x2), dim=1)
x.shape  # (6, 2, 3)
```

You cannot just call custom functions on Placeholders.

An attempt to do it will look like this: `x = Input((1, 2, 3)); x = torch.abs(x);`

```
TypeError: abs(): argument 'input' (position 1) must be Tensor, not Input
```

## References

* [https://www.tensorflow.org/guide/keras/functional](https://www.tensorflow.org/guide/keras/functional)