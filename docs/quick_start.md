# Quick Start

## Features

* Keras-like Functional API
* No overhead during execution
* Supports advanced stuff:
	* Reusing layers
	* Shared layers
	* Multiple inputs and outputs
	* Custom, user-defined modules

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

To register a new layer, e.g. ``nn.Linear`` in Pytorch Symbolic, you have two options:

* `layer(symbolic_tensor)` (like in Keras Symbolic API)
* `symbolic_tensor(layer)` (like nowhere else)

There are no differences between them.
They both return a SymbolicTensor and they create identical models.

**What is a Symbolic Tensor?**

* Think of it as placeholder for your data.
* Use it like `torch.Tensor`

Let's play with it:

```py
print(inputs)
print(inputs + 1)
print(inputs)
```

```stdout
<SymbolicTensor at 0x7f73f2d49ac0; child of 0; parent of 1>
<SymbolicTensor at 0x7f73e3fc4970; child of 1; parent of 0>
<SymbolicTensor at 0x7f73f2d49ac0; child of 0; parent of 2>
```

At first, `inputs` was a parent to `outputs` only.

But when we wrote `inputs + 1`, a new node was registered as
the second child of `inputs`.

Symbolic Tensors have useful attributes.
Using them we can instantly obtain shapes of intermediate outputs,
instead of deriving them by hand. For example:

```py
print(inputs.shape)
print(inputs.features)
```

```stdout
(None, 784)
784
```

Doing this, we:

* Write less code
* Write easier code

## Examples step by step

### Model for RGB images

1. Get your symbolic inputs. You can specify or skip batch size. There are a few ways to do it:
	* `inputs = Input(shape=(C, H, W))`
	* `inputs = Input(shape=(B, C, H, W), batched=False)`
	* `inputs = Input(batch_shape=(B, C, H, W))`
2. Register new modules in the model. There are a few ways:
	* `outputs = inputs(layer)`
	* `outputs = layer(inputs)`
	* `outputs = add_to_model(layer, inputs)`
3. Use standard operations on Symbolic Tensors: `+`, `-`, `*`, `/`, `**` and `abs`:
	* For example, `x=2+inputs` or `x=inputs%y` will work as expected.
4. To concatenate (similarly for stacking) Symbolic Tensors:
	* use `useful_layers.ConcatOpLayer(dim=1)`
	* add custom function to the model:  `add_to_model(torch.concat, (x, y), dim=1)`
5. When working with Symbolic Tensors, use `.shape` property or its handy shortcuts:
	* `.C` and `.channels` equals `.shape[1]` for RGB data
	* `.H` equals `.shape[2]` for RGB data
	* `.W` equals `.shape[3]` for RGB data
	* `.HW` is (height, width) tuple for RGB data
6. Finally, create the model: `model = SymbolicModel(inputs, outputs)`.
7. Use `model` as a normal PyTorch `nn.Module`. It's 100% compatible.

### Sequential topology example

In PyTorch, there's `nn.Sequential` that allows creating simple sequential models.

In Pytorch Symbolic, you can do it as well, but it is *much* more flexible.

```python
from torch import nn
from pytorch_symbolic import Input, SymbolicModel

x = inputs = Input((3, 128, 128))

x = nn.Conv2d(in_channels=x.channels, out_channels=16, kernel_size=3)(x)
x = nn.MaxPool2d(kernel_size=2)(x)(nn.ReLU())

x = nn.Conv2d(in_channels=x.channels, out_channels=32, kernel_size=3)(x)
x = nn.MaxPool2d(kernel_size=2)(x)(nn.ReLU())

x = nn.Conv2d(in_channels=x.channels, out_channels=64, kernel_size=3)(x)
x = nn.MaxPool2d(kernel_size=2)(x)(nn.ReLU())

x = nn.Conv2d(in_channels=x.channels, out_channels=64, kernel_size=3)(x)
x = nn.MaxPool2d(kernel_size=2)(x)(nn.ReLU())(nn.Flatten())

outputs = nn.Linear(in_features=x.features, out_features=10)(x)
model = SymbolicModel(inputs=inputs, outputs=outputs)
assert model.output_shape == (None, 10)
```

### Multiple inputs example

There's nothing stopping you from using multiple input and output nodes.

```python
from torch import nn
from pytorch_symbolic import Input, SymbolicModel

task1_input = x = Input(shape=(3, 32, 32))
task2_input = y = Input(shape=(64,))

x = x(nn.Conv2d(x.channels, 16, 3))
x = x(nn.MaxPool2d(3))(nn.ReLU())(nn.Flatten())
head1_out = x(nn.Linear(x.features, 200))

y = y(nn.Linear(y.features, 512))(nn.ReLU())
y = y(nn.Linear(y.features, 512))(nn.ReLU())
head2_out = y(nn.Linear(y.features, 200))

x = head1_out + head2_out  # elementwise sum
x = x(nn.Linear(x.features, 400))(nn.ReLU())
task1_out = x(nn.Linear(x.features, 10))
task2_out = x(nn.Linear(x.features, 1))

model = SymbolicModel((task1_input, task2_input), (task1_out, task2_out))
```

You can use this model in following way:

```py
import torch

data1 = torch.rand(16, 3, 32, 32)
data2 = torch.rand(16, 64)

outs1, outs2 = model(data1, data2)
```

## Special cases

### Use custom functions on Symbolic Tensors

If you want to use function from `torch.nn.functional` or basically
any custom function in your model, you have a few options.
The recommended way is to wrap it in a `nn.Module`.

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

We already wrapped a few useful functions for you, e.g. `concat` and `stack`.

They are available in `pytorch_symbolic.useful_layers`.

#### Alternative for custom functions

If you really hate classes or you are in a hurry, we got you covered.

You can register almost any function in your Symbolic Model:

```python
import torch
from pytorch_symbolic import Input
from pytorch_symbolic.functions_utility import add_to_model

x1 = Input(shape=(3, 3))
x2 = Input(shape=(5, 3))
x = add_to_model(torch.concat, (x1, x2), dim=1)
x.shape  # (None, 8, 2, 3)
```

But if you try this: `x = torch.abs(x)` you will get:

```
TypeError: abs(): argument 'input' (position 1) must be Tensor, not Input
```

You need to use `add_to_model` like `add_to_model(torch.abs, x)`.

### Modules with multiple inputs

The best way to use modules with multiple inputs is to use `layer(*args)` notation.

Just pass multiple arguments to the layer:

```python
from pytorch_symbolic import Input, useful_layers

x1 = Input(shape=(1, 2, 3))
x2 = Input(shape=(5, 2, 3))
x = useful_layers.ConcatLayer(dim=1)(x1, x2)
x.shape  # (None, 6, 2, 3)
```

Alternatively, using the other notation, do it like this `symbolic_tensor(layer, *other_args)`:

```python
from pytorch_symbolic import Input, useful_layers

x1 = Input(shape=(1, 2, 3))
x2 = Input(shape=(5, 2, 3))
x = x1(useful_layers.ConcatLayer(dim=1), x2)
x.shape  # (None, 6, 2, 3)
```

## References

* [https://www.tensorflow.org/guide/keras/functional](https://www.tensorflow.org/guide/keras/functional)
