# Pytorch Symbolic

[//]: # (To get badges go to https://shields.io/ and use https://pypi.org/pypi/slicemap/json as data url. Query fields using dot as the separator.)

[![PyPi version](https://img.shields.io/badge/dynamic/json?label=latest&query=info.version&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpytorch-symbolic%2Fjson)](https://pypi.org/project/pytorch-symbolic)
[![PyPI license](https://img.shields.io/badge/dynamic/json?label=license&query=info.license&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpytorch-symbolic%2Fjson)](https://pypi.org/project/pytorch-symbolic)
[![Documentation Status](https://readthedocs.org/projects/pytorch-symbolic/badge/?version=latest)](https://pytorch-symbolic.readthedocs.io/en/latest/?badge=latest)
[![Notebook](https://github.com/gahaalt/pytorch-symbolic/actions/workflows/notebook.yaml/badge.svg)](https://github.com/gahaalt/pytorch-symbolic/actions/workflows/notebook.yaml)
[![Python 3.7](https://github.com/gahaalt/pytorch-symbolic/actions/workflows/python-3.7.yaml/badge.svg)](https://github.com/gahaalt/pytorch-symbolic/actions/workflows/python-3.7.yaml)
[![Python 3.10](https://github.com/gahaalt/pytorch-symbolic/actions/workflows/python-3.10.yaml/badge.svg)](https://github.com/gahaalt/pytorch-symbolic/actions/workflows/python-3.10.yaml)

Pytorch Symbolic is MIT licensed library that adds symbolic API for model creation to PyTorch.

Pytorch Symbolic makes it easier and faster to define complex models.
It spares you writing boilerplate code.
It aims to be PyTorch equivalent for [Keras Functional API](https://keras.io/guides/functional_api/).

Features:

* Small extension of PyTorch
* No dependencies besides PyTorch
* Produces models entirely compatible with PyTorch
* Overhead free as tested in [benchmarks](benchmarks.md)
* Reduces the amount of boilerplate code
* Works well with complex architectures
* Code and documentation is automatically tested

## Example

To create a symbolic model, you need Symbolic Tensors and `torch.nn.Module`.
Register layers and operations in your model by calling ``layer(inputs)`` or
equivalently ``inputs(layer)``.
Layers will be automagically added to your model and
all operations will be replayed on the real data.
That's all!

Using Pytorch Symbolic, we can define a working classifier in a few lines of code:

```python
from torch import nn
from pytorch_symbolic import Input, SymbolicModel

inputs = Input(shape=(1, 28, 28))
x = nn.Flatten()(inputs)
x = nn.Linear(x.shape[1], 10)(x)(nn.Softmax(1))
model = SymbolicModel(inputs=inputs, outputs=x)
model.summary()
```

```stdout
_______________________________________________________
     Layer       Output shape        Params   Parent   
=======================================================
1    Input_1     (None, 1, 28, 28)   0                 
2    Flatten_1   (None, 784)         0        1        
3    Linear_1    (None, 10)          7850     2        
4*   Softmax_1   (None, 10)          0        3        
=======================================================
Total params: 7850
Trainable params: 7850
Non-trainable params: 0
_______________________________________________________
```

**See more examples
in [Documentation Quick Start](https://pytorch-symbolic.readthedocs.io/en/latest/quick_start/).**

## Gentle introduction

There's a jupyter notebook showing the basic usage of Pytorch Symbolic. With it you will:

* Learn Pytorch Symbolic in an interactive way
* Try the package before installing it on your computer
* See visualizations of graphs that are created under the hood

Click:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gahaalt/pytorch-symbolic/blob/develop/gentle-introduction.ipynb)

## Installation

Install Pytorch Symbolic easily with pip:

```bash
pip install pytorch-symbolic
```

## Links

* [See Documentation](https://pytorch-symbolic.readthedocs.io/en/latest/quick_start)
* [See on GitHub](https://github.com/gahaalt/pytorch-symbolic/)
* [See on PyPI](https://pypi.org/project/pytorch-symbolic/)

Create an issue if you noticed a problem!

Send me an e-mail if you want to get involved: [sjmikler@gmail.com](mailto:sjmikler@gmail.com).
