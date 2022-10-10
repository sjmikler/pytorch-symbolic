# Accelerating Networks

## CUDA Graphs

CUDA Graphs can greatly speed up your models by reducing GPU idle time
and removing the overhead caused by CPU making GPU calls.

Pytorch Functional comes with easy to use CUDA Graphs support.

To enable it, use:

```py
...
model = FunctionalModel(inputs, outputs, enable_cuda_graphs=True)
```

When using CUDA Graphs, please remember to cast inputs to GPU.
