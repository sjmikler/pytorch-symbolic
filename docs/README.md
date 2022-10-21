# Pytorch Symbolic

[//]: # (To get badges go to https://shields.io/ and use https://pypi.org/pypi/slicemap/json as data url. Query fields using dot as the separator.)

[![PyPi version](https://img.shields.io/badge/dynamic/json?label=latest&query=info.version&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpytorch-symbolic%2Fjson)](https://pypi.org/project/pytorch-symbolic)
[![PyPI license](https://img.shields.io/badge/dynamic/json?label=license&query=info.license&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpytorch-symbolic%2Fjson)](https://pypi.org/project/pytorch-symbolic)
[![Documentation Status](https://readthedocs.org/projects/pytorch-symbolic/badge/?version=latest)](https://pytorch-symbolic.readthedocs.io/en/latest/?badge=latest)
[![Notebook](https://github.com/gahaalt/pytorch-symbolic/actions/workflows/notebook.yaml/badge.svg)](https://github.com/gahaalt/pytorch-symbolic/actions/workflows/notebook.yaml)
[![Python 3.7](https://github.com/gahaalt/pytorch-symbolic/actions/workflows/python-3.7.yaml/badge.svg)](https://github.com/gahaalt/pytorch-symbolic/actions/workflows/python-3.7.yaml)
[![Python 3.10](https://github.com/gahaalt/pytorch-symbolic/actions/workflows/python-3.10.yaml/badge.svg)](https://github.com/gahaalt/pytorch-symbolic/actions/workflows/python-3.10.yaml)

Pytorch Symbolic is MIT licensed library that adds symbolic API for model creation to PyTorch.

Defining complex models in PyTorch requires creating classes and writing boilerplate code.

[Defining models in Keras is easier](https://www.tensorflow.org/guide/keras/symbolic).
Pytorch Symbolic makes it just as easy.


Features:

* Small extension of PyTorch
* No dependencies besides PyTorch
* Produces models entirely compatible with PyTorch
* Overhead free, tested in [benchmarks](https://pytorch-symbolic.readthedocs.io/en/latest/benchmarks/)
* Reduces the amount of code you write
* Works well with complex architectures
* Code and documentation is automatically tested

## Example

To create a symbolic model, you'll use symbolic tensors and nn.Modules.
Add layers by calling ``layer(symbolic_tensor)`` or equivalently ``symbolic_tensor(layer)``. 
That's all!

Create symbolic model just like in Keras:
treat symbolic tensors as if they were normal tensors.

Layers will be automagically registered in your model.

```python
from torch import nn
from pytorch_symbolic import Input

x = Input(shape=(1, 28, 28))  # Input is a SymbolicTensor
print(x)

x = nn.Flatten()(x)  # Every layer operation returns another SymbolicTensor
x = x(nn.Flatten())  # This is equivalent to previous line
print(x)
```

```
<SymbolicTensor at 0x7faf5779d130; child of 0; parent of 0>
<SymbolicTensor at 0x7fafea899f10; child of 1; parent of 0>
```

Using symbolic tensors, we can define a working classificator in a few lines of code:

```python
from torch import nn
from pytorch_symbolic import SymbolicModel, Input

inputs = Input(shape=(1, 28, 28))
x = nn.Flatten()(inputs)
x = nn.Linear(x.shape[1], 10)(x)(nn.ReLU())
model = SymbolicModel(inputs=inputs, outputs=x)
model
```

```
SymbolicModel(
  (module0_depth1): Flatten(start_dim=1, end_dim=-1)
  (module1_depth2): Linear(in_features=784, out_features=10, bias=True)
  (module2_depth3): ReLU()
)
```

**See more examples in [Quick Start](https://pytorch-symbolic.readthedocs.io/en/latest/quick_start/).**

## Gentle introduction

There's a notebook showing the basic usage of Pytorch Symbolic. You will:

* Learn Pytorch Symbolic in an interactive way
* Try the package before installing it on your computer
* See visualizations of graphs that are createdunder the hood

Click:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gahaalt/pytorch-symbolic/blob/develop/gentle-introduction.ipynb)

## Installation

Install easily with pip:

```
pip install pytorch-symbolic
```

## Links

* [See Documentation](https://pytorch-symbolic.readthedocs.io/)
* [See on GitHub](https://github.com/gahaalt/pytorch-symbolic/)
* [See on PyPI](https://pypi.org/project/pytorch-symbolic/)

Create an issue if you noticed a problem!
Send me an e-mail if you want to get involved: [sjmikler@gmail.com](mailto:sjmikler@gmail.com).
