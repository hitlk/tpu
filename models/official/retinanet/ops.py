import time

import tensorflow as tf
from tensorflow import layers

def group_norm(x, G=32, esp=1e-5, name=None):
  """Group normalization."""
  with tf.name_scope(name, 'group_norm', values=[x]):
    x = tf.transpose(x, [0, 3, 1, 2])
    N, C, H, W = x.get_shape().as_list()
    G = min(G, C)
    x = tf.reshape(x, [N, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + esp)

    # per channel gamma and beta
    beta = tf.get_variable('beta', [C], initializer=tf.constant_initializer(0.0))
    gamma = tf.get_variable('gamma', [C], initializer=tf.constant_initializer(1.0))

    beta = tf.reshape(beta, [1, C, 1, 1])
    gamma = tf.reshape(gamma, [1, C, 1, 1])
    output = tf.reshape(x, [N, C, H, W]) * gamma + beta
    output = tf.transpose(output, [0, 3, 1, 2])

    return output
