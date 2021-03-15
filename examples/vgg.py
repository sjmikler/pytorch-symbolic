from torch import nn
from pytorch_functional import Input, FunctionalModel, layers


def classifier(flow, n_classes, pooling="avgpool"):
    if pooling == 'catpool':
        maxp = flow(nn.MaxPool2d(kernel_size=(flow.H, flow.W)))
        avgp = flow(nn.AvgPool2d(kernel_size=(flow.H, flow.W)))
        flow = maxp(layers.ConcatOpLayer(dim=1), avgp)(nn.Flatten())
    if pooling == 'avgpool':
        flow = flow(nn.AvgPool2d(kernel_size=(flow.H, flow.W)))(nn.Flatten())
    if pooling == 'maxpool':
        flow = flow(nn.MaxPool2d(kernel_size=(flow.H, flow.W)))(nn.Flatten())
    return flow(nn.Linear(flow.features, n_classes))


def VGG(input_shape,
        n_classes,
        version=None,
        group_sizes=(1, 1, 2, 2, 2),
        channels=(64, 128, 256, 512, 512),
        pools=(2, 2, 2, 2, 2),
        activation=nn.ReLU(),
        final_pooling="avgpool",
        **kwargs):
    if kwargs:
        print(f"VGG: unknown parameters: {kwargs.keys()}")
    if version:
        if version == 11:
            group_sizes = (1, 1, 2, 2, 2)
        elif version == 13:
            group_sizes = (2, 2, 2, 2, 2)
        elif version == 16:
            group_sizes = (2, 2, 3, 3, 3)
        elif version == 19:
            group_sizes = (2, 2, 4, 4, 4)
        else:
            raise NotImplementedError(f"Unkown version={version}!")

    inputs = Input(shape=input_shape)
    flow = inputs

    iteration = 0
    for group_size, width, pool in zip(group_sizes, channels, pools):
        if iteration == 0:
            iteration = 1
        else:
            flow = flow(nn.MaxPool2d(pool))

        for _ in range(group_size):
            flow = flow(nn.Conv2d(flow.channels, width, 3, 1, 1, bias=False))
            flow = flow(nn.BatchNorm2d(flow.features))(activation)

    outs = classifier(flow, n_classes, final_pooling)
    model = FunctionalModel(inputs=inputs, outputs=outs)
    return model


if __name__ == "__main__":
    import torch
    from pytorch_functional import tools
    from logging import basicConfig, DEBUG

    basicConfig(level=DEBUG)

    model = VGG(
        input_shape=(3, 32, 32),
        n_classes=10,
        version=11
    )

    input = torch.rand(1, 3, 32, 32)
    outs = model.forward(input)
    print(f"Parameters: {tools.get_parameter_count(model)}")
