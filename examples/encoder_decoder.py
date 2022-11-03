#  Copyright (c) 2022 Szymon Mikler

"""
CORRESPONDES TO THE EXAMPLE FROM TENSORFLOW GUIDE
https://www.tensorflow.org/guide/keras/functional
"""

from torch import nn

from pytorch_symbolic import Input, SymbolicModel, useful_layers


def simple_encoder_decoder(input_shape=(1, 28, 28)):
    relu = nn.ReLU()

    encoder_input = x = Input(shape=input_shape)
    x = x(nn.Conv2d(x.channels, 16, 3))(relu)
    x = x(nn.Conv2d(x.channels, 32, 3))(relu)
    x = x(nn.MaxPool2d(3))
    x = x(nn.Conv2d(x.channels, 32, 3))(relu)
    x = x(nn.Conv2d(x.channels, 16, 3))(relu)
    encoder_output = x(nn.MaxPool2d(kernel_size=(x.H, x.W)))(nn.Flatten())
    encoder = SymbolicModel(inputs=encoder_input, outputs=encoder_output)

    decoder_input = x = Input(shape=(encoder_output.features,))
    x = x(useful_layers.ReshapeLayer((1, 4, 4)))
    x = x(nn.ConvTranspose2d(x.channels, 16, 3))(relu)
    x = x(nn.ConvTranspose2d(x.channels, 32, 3))(relu)
    x = x(nn.Upsample(3))
    x = x(nn.ConvTranspose2d(x.channels, 16, 3))(relu)
    decoder_output = x(nn.ConvTranspose2d(x.channels, 1, 3))(relu)
    decoder = SymbolicModel(inputs=decoder_input, outputs=decoder_output)

    autoencoder_input = Input(shape=input_shape)
    encoded_img = autoencoder_input(encoder)
    decoded_img = encoded_img(decoder)
    autoencoder = SymbolicModel(autoencoder_input, decoded_img)
    return autoencoder
