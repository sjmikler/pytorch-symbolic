# Advanced stuff

## Underlying graphs

Deep neural networks can be represented as directed acyclic graphs (DAGs) where:

* nodes are inputs, outputs or intermediate states
* edges are layers (in general transformations or functions)

In such graph,
there exists a nonempty set of input nodes and a nonempty set of output nodes.
If your architecture meets the above conditions, it can be created in a symbolic manner.

Every time you create an `Input`, you create a node in the graph.

```python
from pytorch_symbolic import Input

x = Input(shape=(1,))
print(x)
```

```stdout
<Input at 0x7f127158fd60; 0 parents; 0 children>
```

`Input` is inheriting from `SymbolicTensor` and there's nothing different about using them,
besides how you create them.
In fact, you'll be able to use non-input `SymbolicTensor` as an input to your model.

Three rules you should not break:

1. `Input` has no parents 
2. every other `SymbolicTensor` has at least one parent
3. every other `SymbolicTensor` is a result of operation on `Input` or `SymbolicTensor`

The provided public API will not let you break these, 
but if you are really determined, you can break them by modifying private variables.
This can lead to directed cycles in your graph! If this happens, you won't be able to
create the model from your graph, instead you'll be attacked with an assertion error.

One exception is when your graph has directed cycles,
but you create the model from a subgraph
(all nodes between `inputs` and `outputs` induce the subgraph)
that doesn't contain any directed cycle. Then you'll be fine and the model will be created.

In general, when you transform any `SymbolicTensor`, a new `SymbolicTensor` is created.

```py
y = x + 2
print(y)
```

```stdout
<SymbolicTensor at 0x7f121e11adf0; 1 parents; 0 children>
```

This _transformation_ is always just an `nn.Module`. Even when you use `+` or `-`.

These operators are just handy shortcuts for a module that adds two inputs.

```py
print(y.layer)
```

```stdout
AnyOpLayer()
```

When you add new nodes to the graph,
they will be registered as children of other nodes.
You can check children or parents of every node. For each node, you can access:

* `node.parents: Tuple[SymbolicTensor, ...]`
* `node.children: Tuple[SymbolicTensor, ...]`

It's impossible to modify parents after `SymbolicTensor` is created,
but new children can always be added.
Every operation on `SymbolicNode` creates at least one new child.
If your `nn.Module` has more than one output, more children will be created.
Usually modules have just one output:

```py
print(y)
a = y - 1
print(y)
b = y / 2
print(y)
```

```stdout
<SymbolicTensor at 0x7f121e11adf0; 1 parents; 0 children>
<SymbolicTensor at 0x7f121e11adf0; 1 parents; 1 children>
<SymbolicTensor at 0x7f121e11adf0; 1 parents; 2 children>
```

We created a bunch of children for `y`. Let's see them:

```py
print(y.children)
```

```stdout
(<SymbolicTensor at 0x7f121e107d00; 1 parents; 0 children>,
 <SymbolicTensor at 0x7f121e107760; 1 parents; 0 children>)
```

But when working with large graphs, it might not be very fun to inspect
them by printing children and parents to stdout. There is a nicer alternative though.

Pytorch Symbolic provides a basic graph drawing utility.
It will work only if you have optional dependencies installed:

* networkx
* matplotlib
* scipy

You can install them manually or using `pip install pytorch-symbolic[full]`.

> If you don't have optional dependencies, you can use the package without drawing utility.

Let us draw our graph:

```py
from pytorch_symbolic import graph_algorithms

graph_algorithms.draw_graph(inputs=x, node_text_namespace=globals())
```

![images/draw_graph1.png](images/draw_graph1.png)

We use `node_text_namespace=globals()` so that Pytorch Symbolic attempts
to display variable names on nodes.
This is nice for understanding what is going on in the graph.
It can be more difficult, e.g. when graph was defined in a local namespace.
Instead, you can use `node_text_func` to define custom labels on nodes.
This argument is a `Callable` that takes `SymbolicTensor` as input and returns `str` as output. By default, it displays tensor shape.

Be careful! Drawing utility is not very refined,
so it might not work well for large neural networks.
You can tune the figure size and other stuff using matplotlib:

```py
import matplotlib.pyplot as plt

nodes = [a + i for i in range(10)]
out = sum(nodes)

plt.figure(figsize=(10, 10), constrained_layout=True)

graph_algorithms.draw_graph(
    inputs=x,
    outputs=out,
    edge_text_func=lambda x: "",
    node_text_namespace=globals(),
)
```

![images/draw_graph2.png](images/draw_graph2.png)

Use `edge_text_func` to display custom labels on the edges.
By default it is the name of the layer, e.g. `Linear` or `Conv2d`.

## Creating models

After creating Symbolic Tensors and defining the operations,
you want to enclose them in a model.
It might be a neural network or just an arbitrary graph of computations.
After creating `SymbolicModel` you will be able to use it with your own data.

```python
from torch import nn
from pytorch_symbolic import Input, SymbolicModel

inputs = x = Input((784,))

for _ in range(3):
    x = nn.Linear(x.features, 100)(x)

x = nn.Linear(x.features, 10)(x)
model = SymbolicModel(inputs, x)
```

You can use the plotting utility directly on your model:

```py
from pytorch_symbolic import graph_algorithms

graph_algorithms.draw_graph(model=model)
```

![images/draw_graph3.png](images/draw_graph3.png)

## Multiple inputs and outputs

If your `nn.Module` has multiple inputs or outputs, that's fine.

```python
from torch import nn
from pytorch_symbolic import Input, SymbolicModel, useful_layers, graph_algorithms

add_n = useful_layers.AnyOpLayer(op=lambda *args: sum(args))

inputs = [Input((5,)) for _ in range(5)]
intermediate = add_n(*inputs)
outputs = [intermediate / i for i in range(1, 5)]

model = SymbolicModel(inputs, outputs)
graph_algorithms.draw_graph(model=model)
```

![images/draw_graph4.png](images/draw_graph4.png)

Notice that we used custom `add_n` module instead
of just `intermediate = sum(inputs)`.
Both versions would be correct.
In fact, they produce equivalent models, but their underlying graphs are different.

Let us compare two examples:

```py
inputs = [Input((5,)) for _ in range(3)]
graph_algorithms.draw_graph(inputs=inputs, outputs=add_n(*inputs))
```

![images/draw_graph5.png](images/draw_graph5.png)

And:

```
graph_algorithms.draw_graph(inputs=inputs, outputs=sum(inputs))
```

![images/draw_graph6.png](images/draw_graph6.png)
This is because `sum` is executing multiple `+ / __add__` operations under the hood.

## Reusing existing layers

Each node except input is associated with some `nn.Module`.
When you are reusing a part of the graph, you are reusing all underlying `nn.Module` too.
In fact, you can have multiple models sharing the same weights.
Or one model reusing the same weights multiple times.

For example, imagine you created a model that classifies RGB images:

```python
from torch import nn
from pytorch_symbolic import Input, SymbolicModel

inputs = x = Input((1, 28, 28))

for _ in range(3):
    x = nn.Conv2d(x.C, 8, 3, padding=1)(x)(nn.ReLU())

features = x

x = nn.Flatten()(x)
outputs = nn.Linear(x.features, 10)(x)

classifier = SymbolicModel(inputs, outputs)
print(classifier.output_shape)
```

```stdout
(None, 10)
```

After training `classifier`, you might decide that you want to inspect the intermediate features.

```py
feature_extractor = SymbolicModel(inputs, features)
print(feature_extractor.output_shape)
```

```stdout
(None, 8, 28, 28)
```

This model, `feature_extractor`, uses the same underlying weights
as already trained `classifier`,
but it outputs the intermediate features you wanted to inspect.
It does so without modifying the original model!

There's also another way of achieving the end result, this one modifies the original model:

```py
classifier.add_output(features)
print(classifier.output_shape)
```

```stdout
((None, 10), (None, 8, 28, 28))
```

## Nested models

Instance of `SymbolicModel` is just an `nn.Module` and you can use it as such.
This means you can use it anywhere, including another `SymbolicModel` or vanilla model.
Create new models, using the existing ones.
Here we create a model that calculates how similar are two feature maps generated 
by previously defined `feature_extractor`:

```py
import torch
from pytorch_symbolic import add_to_graph, graph_algorithms

inputs1 = Input((1, 32, 32))
inputs2 = Input((1, 64, 64))

noise = Input((1, 64, 64))

features1 = feature_extractor(inputs1)
features2 = feature_extractor(inputs2 + noise)
features2 = nn.MaxPool2d(2)(features2)

diffs = (features1 - features2) ** 2

outputs = add_to_graph(torch.sum, diffs, dim=(1, 2, 3))
strange_model = SymbolicModel((inputs1, inputs2, noise), outputs)

graph_algorithms.draw_graph(model=strange_model)
```

![images/draw_graph7.png](images/draw_graph7.png)

## Recreate Toy ResNet

We took an example of a toy ResNet from
[tensorflow guide](https://www.tensorflow.org/guide/keras/symbolic) and
recreated it in a few different ways. Note that their example is **16 lines of code long**,
excluding imports and utilities.

> In [benchmarks](benchmarks.md) you can see this model benchmarked!

Using Pytorch Symbolic, you can create toy ResNet using exactly as many lines as using Keras:

```python
from torch import nn
from pytorch_symbolic import Input, SymbolicModel

inputs = Input(shape=(3, 32, 32))
x = nn.Conv2d(inputs.C, 32, 3)(inputs)(nn.ReLU())
x = nn.Conv2d(x.C, 64, 3)(x)(nn.ReLU())
block_1_output = nn.MaxPool2d(3)(x)

x = nn.Conv2d(block_1_output.C, 64, 3, padding=1)(block_1_output)(nn.ReLU())
x = nn.Conv2d(x.C, 64, 3, padding=1)(x)(nn.ReLU())
block_2_output = x + block_1_output

x = nn.Conv2d(block_2_output.C, 64, 3, padding=1)(block_2_output)(nn.ReLU())
x = nn.Conv2d(x.C, 64, 3, padding=1)(x)(nn.ReLU())
block_3_output = x + block_2_output

x = nn.Conv2d(x.C, 64, 3)(block_3_output)(nn.ReLU())
x = nn.AvgPool2d(kernel_size=x.HW)(x)(nn.Flatten())
x = nn.Linear(x.features, 256)(x)(nn.ReLU())
x = nn.Dropout(0.5)(x)
outputs = nn.Linear(x.features, 10)(x)

model = SymbolicModel(inputs, outputs)
```

In fact, every line of code in Pytorch Symbolic and Keras is functionally equivalent.

For example this line in Keras:

```python
... = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
```

is equivalent to this line in Pytorch Symbolic:

```python
... = nn.Conv2d(x.C, 64, 3, padding=1)(x)(nn.ReLU())
```

Let's analyze what happens in the above lines:

* `nn.Conv2d` is PyTorch equivalent of Keras `layers.Conv2d` layer
* Input channels:
	* In Keras we don't pass them openly - they'll be calculated automatically from the inputs
	* In Pytorch Symbolic we also calculate them automatically using `x.C`, but we pass them openly as an argument
* In both frameworks `64, 3` are the number of output channels and the size of the kernel
* Padding:
	* We use `padding="same"` in Keras
	* We use `padding=1` in PyTorch (for kernel size 3)
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

This took 16 lines of code as well.

You can register new layers in whichever way you want or you can mix them.

### Vanilla PyTorch

A usual way to define a model in PyTorch is to create a class that inherits from `nn.Module`.

PyTorch non-symbolic example of toy ResNet from previous section:

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
