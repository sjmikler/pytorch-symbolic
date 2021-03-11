> Early version of the repo

Defining models in tensorflow is easy: https://www.tensorflow.org/guide/keras/functional \
This makes it just as easy in PyTorch.

# Functional API for model creation

Deep learning models can be often presented as directed acyclic graphs with intermediate outputs as nodes and layers (aka. transformations, functions) as edges. In this graph, there exists a nonempty set of input nodes, which in fact are nodes without any predecessors. Also, there exists a nonempty set of output nodes, which are nodes without any successors. 

If your neural network meets the above conditions, it can be created in a functional manner.

TensorFlow functional example (toy ResNet from https://www.tensorflow.org/guide/keras/functional):
```
import tensorflow as tf
from tensorflow.keras import layers

inputs = tf.keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation="relu")(inputs)
x = layers.Conv2D(64, 3, activation="relu")(x)
block_1_output = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_2_output = layers.add([x, block_1_output])

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_3_output = layers.add([x, block_2_output])

x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10)(x)

model = tf.keras.Model(inputs, outputs)
```
This took 16 lines of code.

# Model definition in PyTorch

An usual way to define a model in PyTorch is an objective one. Steps:
1. define a class that inherits from `nn.Module`
2. define all the layers, including their input shapes, in `__init__` method
3. define the order in which layers are used in `forward` method

The separation of steps 2 and 3 makes network creation more difficult than it should be. Why?
* We have to know the exact shape of the input for each layer, sometimes non-trivial, e.g. if we use strides
* In case of complicated networks, we create the model virtually twice: in `__init__` and in `forward`

PyTorch non-functional example (toy ResNet equivalent):
```
from torch import nn
from torch.nn import functional as F


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

# Advantages of Functional API

In functional API, we create the neural network naturally, as we would create a graph: node by node. Instead of defining layers in one place just to decide later how to connect them, we can do it all at once. For example, with functional API, after creating an input node and a layer, we can instantly tell what shape the output of that layer is and use this shape for creating next layers.

Doing this, we:
* Write less code
* Write easier code


PyTorch functional example (exact equivalent of toy ResNet):
```
from torch import nn
from pytorch_functional import Input, FunctionalModel

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

model = FunctionalModel(inputs, outputs)
```
This took 16 lines of code.

### More functional API examples:
* [simple Encoder-Decoder architecture](examples/encoder_decoder.py) - shows coupling multiple FunctionalModels

# Quick Start

The main difference between tensorflow functional API and `pytorch_functional` is how new layers are registered.
* In TensorFlow you apply `layer` on a placeholder node, like `layer(placeholder) -> placeholder`
* In PyTorch you apply placeholder on a `layer`, like `placeholder(layer) -> placeholder`

They both return the resulting placeholder as an output.


### Creating a functional model:
1. Get input variable placeholder `inputs = Input(shape)`, where `shape` is in `(C, H, W)` format (no batch dimension)
2. Placeholders have useful properties:
    * `.channels` for number of channels
    * `.features` for number of features
    * `.H` for height of the image
    * `.W` for width of the image
    * `.shape` for shape of the intermediate variable, omitting batch dimensions
    * Placeholders have standard operations defined: `+`, `-`, `*`, `/`, `**`, and `abs`
    * Concatenate or stack placeholders using `layers.ConcatOpLayer` and `layers.StackOpLayer`
3. To add a `nn.Module` as a transformation, use: `x_outs = x.apply_layer(l)` or just `x(l)`
4. When all the layers are added, define `my_model = FunctionalModel(inputs, outputs)`
5. Use `my_model` as a normal PyTorch model

### Simple, linear topology example:
```
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
model = FunctionalModel(inputs=inputs, outputs=outputs, assert_output_shape=(10,))
```

### Multiple inputs example (just for showcase, network itself might not make much sense):
```
task1_input = Input(shape=(1, 28, 28))
task2_input = Input(shape=(3, 32, 32))

x = task1_input
x = x(nn.Conv2d(x.channels, 16, 3))
x = x(nn.MaxPool2d(3))(nn.ReLU())
x = x(nn.Flatten())
head1_out = x(nn.Linear(x.features, 200))

x = task2_input
x = x(nn.Conv2d(x.channels, 16, 3))
x = x(nn.MaxPool2d(3))(nn.ReLU())
x = x(nn.Flatten())
head2_out = x(nn.Linear(x.features, 200))

x = head1_out + head2_out
x = x(nn.Linear(x.features, 400))(nn.ReLU())
task1_outputs = x(nn.Linear(x.features, 10))
task2_outputs = x(nn.Linear(x.features, 10))

model = FunctionalModel(inputs=(task1_input, task2_input), outputs=(task1_outputs, task2_outputs))
```

### If a layer takes more than 1 input, pass them with the layer:
```
from pytorch_functional import Input, FunctionalModel, layers

x1 = Input(shape=(5, 6, 7))
x2 = Input(shape=(3, 6, 7))

x = x1(layers.ConcatOpLayer(dim=0), x2)
x.shape # = (8, 6, 7)
```

# Features

- [x] TF like API
- [x] Multiple outputs
- [x] Multiple inputs
- [x] Pruning of unused layers
- [x] Reusing layers option
- [x] Complex topologies
- [ ] Many examples
- [ ] Stability (yes, it is a feature)
- [ ] Built-in graph plotting
- [ ] Non-deterministic graphs

# References

* https://www.tensorflow.org/guide/keras/functional
