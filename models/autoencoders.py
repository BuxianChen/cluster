# -*- coding: UTF-8 -*-
import os

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Reshape
from tensorflow.keras.layers import MaxPool2D, AveragePooling2D, UpSampling2D
from tensorflow.keras import Input, Model

from models.layers import n_conv_norm, n_deconv_norm, dense_norm

MODEL_TYPES = ["mlp", "conv", "allconv"]


def get_mlp_backbones(input_shape, embeded_dim, norm_type, acti_type, *args, **kwargs):
    # input_shape: tuple, embeded_dim: int
    # acti, init = "relu", "he_normal"
    input_dim = 1
    for dim in input_shape:
        input_dim *= dim
    hidden_dim = [500, 500, 2000, embeded_dim]
    # hidden_dim = [embeded_dim]
    encoder_input = Input(shape=input_shape)
    encoder_output = Flatten()(encoder_input)
    encoder_layers = []
    for i in range(len(hidden_dim)-1):
        encoder_layers.extend(dense_norm(units=hidden_dim[i], norm_type=norm_type,
                                         acti_type=acti_type, *args, **kwargs))
    encoder_layers.extend(dense_norm(units=hidden_dim[-1], norm_type=None, acti_type=None, *args, **kwargs))
    for layer in encoder_layers:
        encoder_output = layer(encoder_output)

    decoder_input = Input(shape=(hidden_dim[-1]))
    decoder_layers = []
    for i in range(len(hidden_dim)-2, -1, -1):
        decoder_layers.extend(dense_norm(units=hidden_dim[i], norm_type=norm_type,
                                         acti_type=acti_type, *args, **kwargs))
    decoder_layers.extend(dense_norm(units=input_dim, norm_type=None, acti_type="softmax", *args, **kwargs))
    decoder_output = decoder_input
    for layer in decoder_layers:
        decoder_output = layer(decoder_output)
    decoder_output = Reshape(input_shape)(decoder_output)

    encoder = Model(inputs=encoder_input, outputs=encoder_output, name="mlp_encoder")
    decoder = Model(inputs=decoder_input, outputs=decoder_output, name="mlp_decoder")
    return encoder, decoder

def get_mnist_backbone(embeded_dim, model_type="mlp", norm_type="bn", acti_type="relu"):
    """  For `mnist` and `fashion-mnist` dataset
    Args:
        embeded_dim (int): 10
    Returns:
        encoder: (28, 28, 1) -> (1, 1, 10)
        decoder: (1, 1, 10) -> (28, 28, 1)
    """
    input_shape = (28, 28, 1)
    init = "he_normal"
    if model_type == "mlp":
        return get_mlp_backbones(input_shape, embeded_dim, norm_type, acti_type, kernel_initializer=init)
    elif model_type == "conv":
        # 与原论文略有不同
        acti = acti_type

        encoder_input = Input(shape=input_shape)
        encoder_layers = []
        encoder_layers.extend(n_conv_norm(3, norm_type, acti,
                                          filters=64, kernel_size=3,
                                          padding="valid", kernel_initializer=init))  # 22,22,64
        encoder_layers.append(MaxPool2D(2, 2, padding="valid"))  # 11
        encoder_layers.extend(n_conv_norm(4, norm_type, acti,
                                          filters=128, kernel_size=3,
                                          padding="valid", kernel_initializer=init))  # 3,3,128
        # encoder_layers.append(MaxPool2D(2, 2, padding="valid"))
        encoder_layers.extend(n_conv_norm(1, norm_type, acti,
                                          filters=embeded_dim, kernel_size=1,
                                          padding="valid", kernel_initializer=init))  # 3,3,10
        # encoder_layers.append(AveragePooling2D(2))  # 1, 1, 10
        encoder_layers.append(AveragePooling2D(3))  # 1, 1, 10
        encoder_output = encoder_input
        for layer in encoder_layers:
            encoder_output = layer(encoder_output)

        decoder_input = Input(shape=(1, 1, embeded_dim))
        decoder_layers = []
        decoder_layers.append(UpSampling2D(3))  # 3
        decoder_layers.extend(n_deconv_norm(1, norm_type, acti,
                                            filters=128, kernel_size=1,
                                            padding="valid", kernel_initializer=init))  # 3
        decoder_layers.extend(n_deconv_norm(3, norm_type, acti,
                                            filters=128, kernel_size=3,
                                            padding="valid", kernel_initializer=init))  # 9
        decoder_layers.extend(n_deconv_norm(1, norm_type, acti,
                                            filters=64, kernel_size=3,
                                            padding="valid", kernel_initializer=init))  # 11
        decoder_layers.append(UpSampling2D(2))
        decoder_layers.extend(n_deconv_norm(2, norm_type, acti,
                                            filters=64, kernel_size=3,
                                            padding="valid", kernel_initializer=init))  # 26
        decoder_layers.append(Conv2DTranspose(1, 3, kernel_initializer=init))  # 28, (no bn)
        decoder_output = decoder_input
        for layer in decoder_layers:
            decoder_output = layer(decoder_output)

        encoder = Model(inputs=encoder_input, outputs=encoder_output, name="conv_encoder")
        decoder = Model(inputs=decoder_input, outputs=decoder_output, name="conv_encoder")
        return encoder, decoder
    elif model_type == "allconv":
        acti = acti_type
        encoder_input = Input(shape=input_shape)
        encoder_layers = []
        encoder_layers.extend(n_conv_norm(1, norm_type, acti, filters=32, kernel_size=5, strides=2,
                                          padding="same", kernel_initializer=init))  # 14
        encoder_layers.extend(n_conv_norm(1, norm_type, acti, filters=64, kernel_size=5, strides=2,
                                          padding="same", kernel_initializer=init))  # 7
        encoder_layers.extend(n_conv_norm(1, norm_type, acti, filters=128, kernel_size=3, strides=2,
                                          padding="valid", kernel_initializer=init))  # 3
        encoder_layers.append(Conv2D(embeded_dim, 3, activation="relu", kernel_initializer=init))  # 1
        encoder_output = encoder_input
        for layer in encoder_layers:
            encoder_output = layer(encoder_output)

        decoder_input = Input(shape=(1, 1, embeded_dim))
        decoder_layers = []
        decoder_layers.extend(n_deconv_norm(1, norm_type, acti, filters=128, kernel_size=3,
                                            kernel_initializer=init))  # 3
        decoder_layers.extend(n_deconv_norm(1, norm_type, acti, filters=64, kernel_size=3,
                                            strides=2, padding="valid", kernel_initializer=init))  # 7
        decoder_layers.extend(n_deconv_norm(1, norm_type, acti, filters=32, kernel_size=5,
                                            strides=2, padding="same", kernel_initializer=init))  # 14
        decoder_layers.append(Conv2DTranspose(input_shape[-1], 5, strides=2, padding="same",
                                              activation="relu", kernel_initializer=init))  # 28
        decoder_output = decoder_input
        for layer in decoder_layers:
            decoder_output = layer(decoder_output)
        encoder = Model(inputs=encoder_input, outputs=encoder_output, name="allconvbn_encoder")
        decoder = Model(inputs=decoder_input, outputs=decoder_output, name="allconvbn_decoder")
        return encoder, decoder
    else:
        print('Not defined model: ', model_type)
        exit(0)


def get_cifar_backbone(embeded_dim, model_type="mlp", norm_type="bn", acti_type="relu"):
    """ For `cifar10` and `cifar100` dataset
    Args:
        embeded_dim (int): 10 or 100
    Returns:
        encoder: (32, 32, 3) -> (1, 1, 10) or (1, 1, 100)
        decoder: (1, 1, 10) or (1, 1, 100) -> (32, 32, 3)
    """
    input_shape = (32, 32, 3)
    init = "he_normal"
    if model_type == "mlp":
        return get_mlp_backbones(input_shape, embeded_dim, norm_type, acti_type, kernel_initializer=init)
    elif model_type == "allconv":
        acti = acti_type
        encoder_input = Input(shape=input_shape)
        encoder_layers = []
        encoder_layers.extend(n_conv_norm(1, norm_type, acti, filters=32, kernel_size=5, strides=2,
                                          padding="same", kernel_initializer=init))  # 16
        encoder_layers.extend(n_conv_norm(1, norm_type, acti, filters=64, kernel_size=5, strides=2,
                                          padding="same", kernel_initializer=init))  # 8
        encoder_layers.extend(n_conv_norm(1, norm_type, acti, filters=128, kernel_size=3, strides=2,
                                          padding="valid", kernel_initializer=init))  # 3
        encoder_layers.append(Conv2D(embeded_dim, 3, activation=None, kernel_initializer=init))  # 1
        encoder_output = encoder_input
        for layer in encoder_layers:
            encoder_output = layer(encoder_output)

        decoder_input = Input(shape=(1, 1, embeded_dim))
        decoder_layers = []
        decoder_layers.extend(n_deconv_norm(1, norm_type, acti, filters=128, kernel_size=3,
                                            kernel_initializer=init))  # 3
        decoder_layers.extend(n_deconv_norm(1, norm_type, acti, filters=64, kernel_size=3, output_padding=1,
                                            strides=2, padding="valid", kernel_initializer=init))  # 8
        decoder_layers.extend(n_deconv_norm(1, norm_type, acti, filters=32, kernel_size=5,
                                            strides=2, padding="same", kernel_initializer=init))  # 16
        decoder_layers.append(Conv2DTranspose(input_shape[-1], 5, strides=2, padding="same",
                                              activation="sigmoid", kernel_initializer=init))  # 32
        decoder_output = decoder_input
        for layer in decoder_layers:
            decoder_output = layer(decoder_output)
        encoder = Model(inputs=encoder_input, outputs=encoder_output, name="allconvbn_encoder")
        decoder = Model(inputs=decoder_input, outputs=decoder_output, name="allconvbn_decoder")
        return encoder, decoder
    elif model_type == "conv":
        # 与原文cifar10_2保持一致的encoder结构
        acti = acti_type

        encoder_input = Input(shape=input_shape)
        encoder_layers = []
        encoder_layers.extend(n_conv_norm(3, norm_type, acti,
                                          filters=64, kernel_size=3,
                                          padding="valid", kernel_initializer=init))  # 26,26,64
        encoder_layers.append(MaxPool2D(2, 2, padding="valid"))  # 13
        encoder_layers.extend(n_conv_norm(3, norm_type, acti,
                                          filters=128, kernel_size=3,
                                          padding="valid", kernel_initializer=init))  # 7,7,128
        encoder_layers.append(MaxPool2D(2, 2, padding="valid"))  # 3,3,128
        encoder_layers.extend(n_conv_norm(1, norm_type, acti,
                                          filters=embeded_dim, kernel_size=1,
                                          padding="valid", kernel_initializer=init))  # 3,3,10
        encoder_layers.append(AveragePooling2D(3))
        encoder_output = encoder_input
        for layer in encoder_layers:
            encoder_output = layer(encoder_output)

        decoder_input = Input(shape=(1, 1, embeded_dim))
        decoder_layers = []
        decoder_layers.append(UpSampling2D(3))  # 3
        decoder_layers.extend(n_deconv_norm(1, norm_type, acti,
                                            filters=128, kernel_size=1,
                                            padding="valid", kernel_initializer=init))  # 3
        decoder_layers.extend(n_deconv_norm(1, norm_type, acti,
                                            filters=128, kernel_size=3, strides=2,
                                            padding="valid", kernel_initializer=init))  # 7
        decoder_layers.extend(n_deconv_norm(2, norm_type, acti,
                                            filters=128, kernel_size=3,
                                            padding="valid", kernel_initializer=init))  # 11
        decoder_layers.extend(n_deconv_norm(1, norm_type, acti,
                                            filters=64, kernel_size=3,
                                            padding="valid", kernel_initializer=init))  # 13
        decoder_layers.append(UpSampling2D(2))
        decoder_layers.extend(n_deconv_norm(2, norm_type, acti,
                                            filters=64, kernel_size=3,
                                            padding="valid", kernel_initializer=init))  # 30
        decoder_layers.append(Conv2DTranspose(3, kernel_size=3, kernel_initializer=init))  # 32, (no bn)
        decoder_output = decoder_input
        for layer in decoder_layers:
            decoder_output = layer(decoder_output)

        encoder = Model(inputs=encoder_input, outputs=encoder_output, name="conv_encoder")
        decoder = Model(inputs=decoder_input, outputs=decoder_output, name="conv_encoder")
        return encoder, decoder


# TODO: 以下数据集的骨干网络尚未重构好, 暂不使用
def get_coil100_backbone(embeded_dim, model_type="mlp", norm_type="bn", acti_type="relu"):
    """ For `coil100` dataset
    Args:
        embeded_dim (int): 100
    Returns:
        encoder: (128, 128, 3) -> (1, 1, 100)
        decoder: (1, 1, 100) -> (128, 128, 1)
    """
    input_shape = (128, 128, 3)
    encoder = tf.keras.Sequential()
    encoder.add(tf.keras.layers.Conv2D(32, 5, strides=2, padding="same", activation="relu", input_shape=input_shape))
    encoder.add(tf.keras.layers.Conv2D(64, 5, strides=2, padding="same", activation="relu"))  # 32
    encoder.add(tf.keras.layers.Conv2D(64, 5, strides=2, padding="same", activation="relu"))  # 16
    encoder.add(tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu"))  # 8
    encoder.add(tf.keras.layers.Conv2D(embeded_dim, 8, activation=None))  # 1

    decoder = tf.keras.Sequential()
    decoder.add(tf.keras.layers.Conv2DTranspose(128, 8, activation="relu", input_shape=(1, 1, embeded_dim)))  # 8
    decoder.add(tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu"))  # 16
    decoder.add(tf.keras.layers.Conv2DTranspose(64, 5, strides=2, padding="same", activation="relu"))  # 32
    decoder.add(tf.keras.layers.Conv2DTranspose(32, 5, strides=2, padding="same", activation="relu"))  # 64
    decoder.add(tf.keras.layers.Conv2DTranspose(3, 5, strides=2, padding="same",
                                                activation="sigmoid"))  # 128
    return encoder, decoder


def get_usps_backbone(embeded_dim, model_type="mlp", norm_type="bn", acti_type="relu"):
    """  For `usps` dataset
    Args:
        embeded_dim (int): 10
    Returns:
        encoder: (16, 16, 1) -> (1, 1, 10)
        decoder: (1, 1, 10) -> (16, 16, 1)
    """
    input_shape = (16, 16, 1)
    init = "glorot_uniform"
    if model_type == "all_conv":
        encoder = tf.keras.Sequential()
        encoder.add(tf.keras.layers.Conv2D(32, 5, strides=2, padding="same", activation="relu", input_shape=input_shape))
        encoder.add(tf.keras.layers.Conv2D(64, 5, strides=2, padding="same", activation="relu"))  # 4
        encoder.add(tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu"))  # 2
        encoder.add(tf.keras.layers.Conv2D(embeded_dim, 2, activation="softmax"))  # None

        decoder = tf.keras.Sequential()
        decoder.add(tf.keras.layers.Conv2DTranspose(128, 2, activation="relu", input_shape=(1, 1, embeded_dim)))
        decoder.add(tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu"))
        decoder.add(tf.keras.layers.Conv2DTranspose(32, 5, strides=2, padding="same", activation="relu"))
        decoder.add(tf.keras.layers.Conv2DTranspose(1, 5, strides=2, padding="same",
                                                    activation="sigmoid"))
        return encoder, decoder
    elif model_type == "mlp":
        return get_mlp_backbones(input_shape, embeded_dim, norm_type, acti_type, kernel_initializer=init)


def get_stl10_backbone(embeded_dim, model_type="mlp", norm_type="bn", acti_type="relu"):
    """ For `stl10` dataset
    Args:
        embeded_dim (int): 10
    Returns:
        encoder: (96, 96, 3) -> (1, 1, 10)
        decoder: (1, 1, 10) -> (96, 96, 3)
    """
    input_shape = (96, 96, 3)
    encoder = tf.keras.Sequential()
    encoder.add(tf.keras.layers.Conv2D(32, 5, strides=2, padding="same", activation="relu", input_shape=input_shape))
    encoder.add(tf.keras.layers.Conv2D(64, 5, strides=2, padding="same", activation="relu"))  # 24
    encoder.add(tf.keras.layers.Conv2D(64, 5, strides=2, padding="same", activation="relu"))  # 12
    encoder.add(tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu"))  # 6
    encoder.add(tf.keras.layers.Conv2D(embeded_dim, 6, activation=None))  # 1

    decoder = tf.keras.Sequential()
    decoder.add(tf.keras.layers.Conv2DTranspose(128, 6, activation="relu", input_shape=(1, 1, embeded_dim)))  # 6
    decoder.add(tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu"))  # 12
    decoder.add(tf.keras.layers.Conv2DTranspose(64, 5, strides=2, padding="same", activation="relu"))  # 24
    decoder.add(tf.keras.layers.Conv2DTranspose(32, 5, strides=2, padding="same", activation="relu"))  # 48
    decoder.add(tf.keras.layers.Conv2DTranspose(3, 5, strides=2, padding="same",
                                                activation="sigmoid"))  # 96
    return encoder, decoder


def get_backbone(name, *args, **kwargs):
    available = {"mnist": get_mnist_backbone,  # embeded_dim=10
                 "fashion_mnist": get_mnist_backbone,  # embeded_dim=10
                 "cifar10": get_cifar_backbone,  # embeded_dim=10
                 "cifar100": get_cifar_backbone,  # embeded_dim=100
                 "coil100": get_coil100_backbone,  # embeded_dim=100
                 "usps": get_usps_backbone,  # embeded_dim=10
                 "stl10": get_stl10_backbone, }  # embeded_dim=10
    assert name in available, "Data error"
    return available[name](*args, **kwargs)


def get_pretrain_backbone(name, pretrain_model_dir="results/model"):
    encoder = tf.keras.models.load_model(os.path.join(pretrain_model_dir, name + "_encoder.h5"))
    decoder = tf.keras.models.load_model(os.path.join(pretrain_model_dir, name + "_decoder.h5"))
    return encoder, decoder