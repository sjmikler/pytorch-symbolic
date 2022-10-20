# Benchmarks

Functional API speeds up prototyping and developement process.
But does it sacrifice speed of the model itself?

One of the most important principles in building this library was to avoid this.
It was made with performance in mind.

Standard, class-based model implementation is a baseline for us.
This library aims to be just as fast in all scenarios.

## Hardwarde

Unless stated otherwise, experiments were run on following PC:

```
CPU: i7-12700KF
GPU: RTX 3080 10GB
```

## Thin and extra deep

This is a very thin and deep neural networks with linear layers only.
In such a deep network, if there's overhead in automatically generated `forward` function,
it should be visible here. In larger models, the overhead will be hidden by the layers computation.
This is not a realistic neural network, however.

#### Data

```py
import torch

data = torch.rand(size=(4, 4))
```

#### Model definition

```py
from torch import nn
from pytorch_functional import Input, FunctionalModel

n_layers = 250  # other used values: 500, 750, 1000

x = inputs = Input(shape=(4,))
for _ in range(n_layers):
    x = nn.Linear(4, 4)(x)
model = FunctionalModel(inputs, x)
```

### Inference (cpu)

![images/from_250_to_1000_linear_layers.png](images/from_250_to_1000_linear_layers.png)
> Percentile intervals [25, 75] are visible. Only sequential model seems to be a
> few percents slower than the other two. It is slowing down more, as number of layers is increasing.

## How is `FunctionalModel` optimized?

Functional models reside on a underlying graph structure.
Each `SymbolicTensor` is a node and each layer is an edge that connects two nodes.
Initialy, the forward pass was implemented lazily: by executing `forward` in a layer only when
it was needed by a child node. But such back-calling to the parents creates unecessary overhead.
We are able to precompute the exact order in which the layers needs to be called,
using topological ordering of the underlying graph structure.

Even when we know the order of the layers, there's one more trick.
Accessing structures has significant overhead in Python, so we want to avoid this.
We generate code for `forward` function dynamically, when the model is created.
Thanks to this, `FunctionalModel` executes exactly the same code it would if you
wrote it as a class.

You can even see the generated code yourself:

```python
...
print(model._generated_forward_source)
```

```
def _generated_forward(self,i00,i01):
    l = self._execution_order_layers
    h09 = i01
    for layer in l[0:10]:
        h09 = layer(h09)
    h19 = i00
    for layer in l[10:20]:
        h19 = layer(h19)
    h20 = l[20](h19,h09)
    h21 = l[21](h20)
    h22 = l[22](h21)
    h32 = h22
    for layer in l[23:33]:
        h32 = layer(h32)
    h42 = h22
    for layer in l[33:43]:
        h42 = layer(h42)
    o00 = l[43](h42,h32)
    return o00
```

Additionaly, it's very simple to enable CUDA Graphs
when GPU runtime is available. CUDA Graphs is a novel feature in PyTorch that can greatly
increase the performance of some models.
