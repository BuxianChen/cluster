# -*- coding: UTF-8 -*-
import tensorflow as tf


@tf.function
def kl_dist(y_true, y_pred):
    """ Kullback-Leibler Divergence

    Args:
        y_true (tensor): shape=(batch_size, num_cluster)
        y_pred (tensor): shape=(batch_size, num_cluster)

    Returns:
        loss (tensor): shape=(batch_size,)

    """
    # axis = tf.range(1, tf.keras.backend.ndim(y_true))
    loss = tf.math.reduce_sum(y_true * tf.math.log(y_true / (y_pred + 1e-7) + 1e-7), axis=1)
    return loss


@tf.function
def l2_dist(y_true, y_pred):
    """ mean squared error
    Args:
        y_true (tensor): shape=(batch_size, *other_shape)
        y_pred (tensor): shape=(batch_size, *other_shape)

    Returns:
        loss (tensor): shape=(batch_size,)
    """
    axis = tf.range(1, tf.keras.backend.ndim(y_true))
    loss = tf.math.reduce_mean(tf.math.square(y_true - y_pred), axis=axis)
    return loss


@tf.function
def l1_dist(y_true, y_pred):
    """ mean squared error
    Args:
        y_true (tensor): shape=(batch_size, *other_shape)
        y_pred (tensor): shape=(batch_size, *other_shape)

    Returns:
        loss (tensor): shape=(batch_size,)
    """
    axis = tf.range(1, tf.keras.backend.ndim(y_true))
    loss = tf.math.reduce_mean(tf.math.abs(y_true - y_pred), axis=axis)
    return loss


# deprecated
@tf.function
def dac_loss(y_pred, upper, lower):
    normalized = tf.math.l2_normalize(y_pred, axis=1)
    similarity = tf.matmul(normalized, tf.transpose(normalized))
    u_mask = tf.cast(similarity > upper, similarity.dtype)
    l_mask = tf.cast(similarity < lower, similarity.dtype)
    pos_loss = -u_mask * tf.math.log(tf.clip_by_value(similarity, 1e-10, 1.))
    neg_loss = -l_mask * tf.math.log(tf.clip_by_value(1 - similarity, 1e-10, 1.))
    loss = tf.reduce_mean(pos_loss) + tf.reduce_mean(neg_loss)
    return loss


@tf.function
def js_loss(y_true, y_pred):
    """ JS Divergence

    Args:
        y_true (tensor): shape=(batch_size, num_cluster)
        y_pred (tensor): shape=(batch_size, num_cluster)

    Returns:
        loss (tensor): shape=(batch_size,)

    """
    y = (y_true + y_pred) / 2.
    loss = tf.reduce_sum((y_true * tf.math.log(y_true / y) + y_pred * tf.math.log(y_pred / y)) / 2., axis=1)
    return loss


losses = {"kl": kl_dist, "l1": l1_dist, "l2": l2_dist, "js": js_loss}
