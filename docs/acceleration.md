# Acceleration

## CUDA Graphs

CUDA Graphs can greatly speed up your models by reducing the time when
GPU is idle and removing the overhead caused by CPU making GPU calls.

Pytorch Functional comes with easy to use CUDA Graphs support.

To enable it, use:

```py
...
model = FunctionalModel(inputs, outputs, enable_cuda_graphs=True)
```

If using CUDA Graphs, please make sure your inputs are on the right device.
