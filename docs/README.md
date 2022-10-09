# Pytorch Functional

[//]: # (To get badges go to https://shields.io/ and use https://pypi.org/pypi/slicemap/json as data url. Query fields using dot as the separator.)

[![PyPi version](https://img.shields.io/badge/dynamic/json?label=latest&query=info.version&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpytorch-functional%2Fjson)](https://pypi.org/project/pytorch-functional)
[![PyPI license](https://img.shields.io/badge/dynamic/json?label=license&query=info.license&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpytorch-functional%2Fjson)](https://pypi.org/project/pytorch-functional)
[![Documentation Status](https://readthedocs.org/projects/slicemap/badge/?version=latest)](https://pytorch-functional.readthedocs.io/en/latest/?badge=latest)
[![Python 3.7](https://github.com/gahaalt/pytorch-functional/actions/workflows/python37.yaml/badge.svg)](https://github.com/gahaalt/pytorch-functional/actions/workflows/python37.yaml)
[![Python 3.8](https://github.com/gahaalt/pytorch-functional/actions/workflows/python38.yaml/badge.svg)](https://github.com/gahaalt/pytorch-functional/actions/workflows/python38.yaml)
[![Python 3.9](https://github.com/gahaalt/pytorch-functional/actions/workflows/python39.yaml/badge.svg)](https://github.com/gahaalt/pytorch-functional/actions/workflows/python39.yaml)
[![Python 3.10](https://github.com/gahaalt/pytorch-functional/actions/workflows/python310.yaml/badge.svg)](https://github.com/gahaalt/pytorch-functional/actions/workflows/python310.yaml)

Pytorch Functional is a MIT licensed library that adds functional API for model creation to PyTorch.

Defining complex models in PyTorch requires creating classes.
[Defining models in Keras is easier](https://www.tensorflow.org/guide/keras/functional).
This makes it just as easy in PyTorch. With Pytorch Functional, you can create neural networks
without tedious calculations of input shapes for each layer.

Features:

* Small extension to PyTorch
* No dependencies besides PyTorch
* Produces models entirely compatible with PyTorch
* Reduces the amount of code that you need to write
* Works well with complex architectures

## New in 0.4.0

Using the new version of the API you can create functional model just like in Keras:
by calling the layer with placeholders as arguments. Layer will be then automagically registered in your model.

```py
from torch import nn
from pytorch_functional import FunctionalModel, Input

inputs = Input(shape=(1, 28, 28))
x = nn.Flatten()(inputs)
x = nn.Linear(x.shape[1], 10)(x)
outputs = nn.ReLU()(x)
model = FunctionalModel(inputs, outputs)
model
```

```
FunctionalModel(
    (module000_depth001): Flatten(start_dim=1, end_dim=-1)
    (module001_depth002): Linear(in_features=784, out_features=10, bias=True)
    (module002_depth003): ReLU()
)
```

* 100% backward compatibile models
* You can mix the new and old API
* Works with multiple arguments

## Example

To create a functional model, call a placeholder with the layer as an argument.
This will return another placeholder, which you can use.

```py
from torch import nn
from pytorch_functional import FunctionalModel, Input

inputs = Input(shape=(1, 28, 28))
x = inputs(nn.Flatten())
outputs = x(nn.Linear(x.shape[1], 10))(nn.ReLU())
model = FunctionalModel(inputs, outputs)
model
```

```
FunctionalModel(
    (module000_depth001): Flatten(start_dim=1, end_dim=-1)
    (module001_depth002): Linear(in_features=784, out_features=10, bias=True)
    (module002_depth003): ReLU()
)
```

**See more examples in [Quick Start](https://pytorch-functional.readthedocs.io/en/latest/quick_start/).**

## Installation

Install easily with pip:

```
pip install pytorch-functional
```

## Links

* [See Documentation](https://pytorch-functional.readthedocs.io/)
* [See on GitHub](https://github.com/gahaalt/pytorch-functional/)
* [See on PyPI](https://pypi.org/project/pytorch-functional/)

Create an issue if you have noticed a problem!
Send me an e-mail if you want to get involved: [sjmikler@gmail.com](mailto:sjmikler@gmail.com).
