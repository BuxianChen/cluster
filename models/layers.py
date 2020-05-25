"""
    本文件为神经网络结构的便利函数, 不传递initialize参数
"""
import tensorflow as tf


class Normalization(tf.keras.layers.Layer):
    def __init__(self):
        super(Normalization, self).__init__()
        self.axis = None
        self.eps = 1e-7

    def build(self, input_shape):
        self.axis = tf.range(len(input_shape) - 1)
        self.built = True

    @tf.function
    def call(self, inputs, **kwargs):
        """
        Args:
            inputs (tensor): (B, H, W, C): ndim=3 or (B, C): ndim=1

        Returns:
            outputs (tensor): (B, H, W, C): ndim=3 or (B, C): ndim=1

        """
        mean = tf.math.reduce_mean(inputs, axis=self.axis)
        std = tf.math.reduce_std(inputs, axis=self.axis)
        outputs = (inputs - mean) / (std + self.eps)
        return outputs


normalization_layers = {"bn": tf.keras.layers.BatchNormalization,
                        "normal": Normalization}
activation_layers = {"relu": tf.keras.layers.ReLU,
                     "softmax": tf.keras.layers.Softmax,
                     "linear": tf.keras.layers.Activation("linear")}


# [conv-norm-activate]
def conv_norm(norm_type, acti_type, *args, **kwargs):
    block = [tf.keras.layers.Conv2D(*args, **kwargs)]
    if norm_type is not None:
        block.append(normalization_layers[norm_type]())
    if acti_type is not None:
        block.append(activation_layers[acti_type]())
    return block


# [conv-norm-activate]*n
def n_conv_norm(n, norm_type, acti_type, *args, **kwargs):
    block = []
    for i in range(n):
        block.extend(conv_norm(norm_type, acti_type, *args, **kwargs))
    return block


def deconv_norm(norm_type, acti_type, *args, **kwargs):
    block = [tf.keras.layers.Conv2DTranspose(*args, **kwargs)]
    if norm_type is not None:
        block.append(normalization_layers[norm_type]())
    if acti_type is not None:
        block.append(activation_layers[acti_type]())
    return block


def n_deconv_norm(n, norm_type, acti_type, *args, **kwargs):
    block = []
    for i in range(n):
        block.extend(deconv_norm(norm_type, acti_type, *args, **kwargs))
    return block


# [dense-norm-activate]
def dense_norm(norm_type, acti_type, *args, **kwarg):
    block = [tf.keras.layers.Dense(*args, **kwarg)]
    if norm_type is not None:
        block.append(normalization_layers[norm_type]())
    if acti_type is not None:
        block.append(activation_layers[acti_type]())
    return block
