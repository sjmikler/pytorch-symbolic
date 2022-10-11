# Pytorch Functional

[//]: # (To get badges go to https://shields.io/ and use https://pypi.org/pypi/slicemap/json as data url. Query fields using dot as the separator.)

[![PyPi version](https://img.shields.io/badge/dynamic/json?label=latest&query=info.version&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpytorch-functional%2Fjson)](https://pypi.org/project/pytorch-functional)
[![PyPI license](https://img.shields.io/badge/dynamic/json?label=license&query=info.license&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpytorch-functional%2Fjson)](https://pypi.org/project/pytorch-functional)
[![Documentation Status](https://readthedocs.org/projects/slicemap/badge/?version=latest)](https://pytorch-functional.readthedocs.io/en/latest/?badge=latest)
[![Python 3.7](https://github.com/gahaalt/pytorch-functional/actions/workflows/python37.yaml/badge.svg)](https://github.com/gahaalt/pytorch-functional/actions/workflows/python37.yaml)
[![Python 3.8](https://github.com/gahaalt/pytorch-functional/actions/workflows/python38.yaml/badge.svg)](https://github.com/gahaalt/pytorch-functional/actions/workflows/python38.yaml)
[![Python 3.9](https://github.com/gahaalt/pytorch-functional/actions/workflows/python39.yaml/badge.svg)](https://github.com/gahaalt/pytorch-functional/actions/workflows/python39.yaml)
[![Python 3.10](https://github.com/gahaalt/pytorch-functional/actions/workflows/python310.yaml/badge.svg)](https://github.com/gahaalt/pytorch-functional/actions/workflows/python310.yaml)

Pytorch Functional is MIT licensed library that adds functional API for model creation to PyTorch.

Defining complex models in PyTorch required creating classes.
[Defining models in Keras is easier](https://www.tensorflow.org/guide/keras/functional).
Pytorch Functional makes it just as easy.

With Pytorch Functional, you can create neural networks without tedious calculations of input shape for each layer.

Features:

* Small extension of PyTorch
* No dependencies besides PyTorch
* Produces models entirely compatible with PyTorch
* Reduces the amount of code that you need to write
* Works well with complex architectures
* Package and documentation automatically tested

## Example

To create a functional model, you'll use symbolic tensors and nn.Modules.
You can add new ``nn.Module`` to your model by calling ``module(symbolic_tensor)`` or equivalently ``symbolic_tensor(module)``.

You can create functional model just like in Keras:
by calling the modules and symbolic tensors as if they were normal tensors. Layers will be then automagically registered in your model.

```python
from torch import nn
from pytorch_functional import Input

x = Input(shape=(1, 28, 28))  # Input is a SymbolicTensor
print(x)

x = nn.Flatten()(x)  # Every layer returns another SymbolicTensor
x = x(nn.Flatten())  # This is equivalent
print(x)
```

```
<SymbolicTensor at 0x7faf5779d130; child of 0; parent of 0>
<SymbolicTensor at 0x7fafea899f10; child of 1; parent of 0>
```

Using symbolic tensors, we can define a working classificator in a few lines of code:

```python
from torch import nn
from pytorch_functional import FunctionalModel, Input

inputs = Input(shape=(1, 28, 28))
x = nn.Flatten()(inputs)
x = nn.Linear(x.shape[1], 10)(x)(nn.ReLU())
model = FunctionalModel(inputs=inputs, outputs=x)
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

## Gentle introduction

There's a notebook showing the basic usage of Pytorch Functional. With it you can:

* Learn Pytorch Functional in an interactive way
* See visualizations of graphs that are created under the hood
* Try the package out before installing it on your computer

Click:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gahaalt/pytorch-functional/blob/develop/gentle-introduction.ipynb)

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
