"""
CORRESPONDES TO THE EXAMPLE FROM TENSORFLOW GUIDE
https://www.tensorflow.org/guide/keras/functional
"""

#  Copyright (c) 2022 Szymon Mikler

from torch import nn

from .. import FunctionalModel, Input, layers


def simple_encoder_decoder(input_shape=(1, 28, 28)):
    relu = nn.ReLU()

    encoder_input = x = Input(shape=input_shape)
    x = x(nn.Conv2d(x.channels, 16, 3))(relu)
    x = x(nn.Conv2d(x.channels, 32, 3))(relu)
    x = x(nn.MaxPool2d(3))
    x = x(nn.Conv2d(x.channels, 32, 3))(relu)
    x = x(nn.Conv2d(x.channels, 16, 3))(relu)
    encoder_output = x(nn.MaxPool2d(kernel_size=(x.H, x.W)))(nn.Flatten())
    encoder = FunctionalModel(inputs=encoder_input, outputs=encoder_output)

    decoder_input = x = Input(shape=(encoder_output.features,))
    x = x(layers.ReshapeLayer((1, 4, 4)))
    x = x(nn.ConvTranspose2d(x.channels, 16, 3))(relu)
    x = x(nn.ConvTranspose2d(x.channels, 32, 3))(relu)
    x = x(nn.Upsample(3))
    x = x(nn.ConvTranspose2d(x.channels, 16, 3))(relu)
    decoder_output = x(nn.ConvTranspose2d(x.channels, 1, 3))(relu)
    decoder = FunctionalModel(inputs=decoder_input, outputs=decoder_output)

    autoencoder_input = Input(shape=input_shape)
    encoded_img = autoencoder_input(encoder)
    decoded_img = encoded_img(decoder)
    autoencoder = FunctionalModel(autoencoder_input, decoded_img)
    return autoencoder


if __name__ == "__main__":
    from logging import DEBUG, basicConfig

    import torch

    from pytorch_functional import tools

    basicConfig(level=DEBUG)

    model = simple_encoder_decoder((1, 28, 28))

    inputs = torch.rand(1, 1, 28, 28)
    outs = model.forward(inputs)
    print(f"Parameters: {tools.get_parameter_count(model)}")
