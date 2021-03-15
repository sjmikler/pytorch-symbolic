# EXAMPLE FROM TENSORFLOW GUIDE https://www.tensorflow.org/guide/keras/functional
# TF EQUIVALENT:
"""
encoder_input = keras.Input(shape=(28, 28, 1), name="original_img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

decoder_input = keras.Input(shape=(16,), name="encoded_img")
x = layers.Reshape((4, 4, 1))(decoder_input)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)

decoder = keras.Model(decoder_input, decoder_output, name="decoder")
decoder.summary()

autoencoder_input = keras.Input(shape=(28, 28, 1), name="img")
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = keras.Model(autoencoder_input, decoded_img, name="autoencoder")
autoencoder.summary()
"""

from torch import nn
from pytorch_functional import Input, FunctionalModel, layers


def simple_encoder_decoder(input_shape=(1, 28, 28)):
    relu = nn.ReLU()

    encoder_input = Input(shape=input_shape)
    x = encoder_input
    x = x(nn.Conv2d(x.channels, 16, 3))(relu)
    x = x(nn.Conv2d(x.channels, 32, 3))(relu)
    x = x(nn.MaxPool2d(3))
    x = x(nn.Conv2d(x.channels, 32, 3))(relu)
    x = x(nn.Conv2d(x.channels, 16, 3))(relu)
    encoder_output = x(nn.MaxPool2d(kernel_size=(x.H, x.W)))(nn.Flatten())
    encoder = FunctionalModel(inputs=encoder_input, outputs=encoder_output)

    decoder_input = Input(shape=(encoder_output.features,))
    x = decoder_input(layers.Reshape((1, 4, 4)))
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
    import torch
    from pytorch_functional import tools
    from logging import basicConfig, DEBUG

    basicConfig(level=DEBUG)

    model = simple_encoder_decoder((1, 28, 28))

    input = torch.rand(1, 1, 28, 28)
    outs = model.forward(input)
    print(f"Parameters: {tools.get_parameter_count(model)}")
