import tensorflow as tf
import numpy as np


def dot_or_lookup(features, weights, onehot_input=False):
    if onehot_input:
        return tf.nn.embedding_lookup(weights, features)
    else:
        return tf.matmul(features, weights)


def glorot_variance(shape):
    return 3 / np.sqrt(shape[0] + shape[1])


def make_tf_variable(mean, variance, shape, init="normal"):
    if init == "normal":
        initializer = np.random.normal(mean, variance, size=shape).astype(np.float32)
    elif init == "uniform":
        initializer = np.random.uniform(mean, variance, size=shape).astype(np.float32)

    return tf.Variable(initializer)


def make_tf_bias(shape, init=0):
    if init == 0:
        return tf.Variable(np.zeros(shape).astype(np.float32))
    elif init == 1:
        return tf.Variable(np.ones(shape).astype(np.float32))
