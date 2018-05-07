import time

import tensorflow as tf
from object_detection import shape_utils
from tensorflow import layers

def group_norm(x, G=32, esp=1e-5, name=None):
  """Group normalization."""
  with tf.variable_scope(name, 'group_norm', values=[x]):
    x = tf.transpose(x, [0, 3, 1, 2])
    combined_shape = shape_utils.combined_static_and_dynamic_shape(x)
    N, C, H, W = combined_shape
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
    output = tf.transpose(output, [0, 2, 3, 1])

    return output

def affine_channel(x, name=None):
  """Affine Transformation."""
  with tf.variable_scope(name, 'affine_channel', values=[x]):
    x = tf.transpose(x, [0, 3, 1, 2])
    combined_shape = shape_utils.combined_static_and_dynamic_shape(x)
    N, C, H, W = combined_shape
    scale = tf.get_variable('scale', [C], initializer=tf.constant_initializer(1.0))
    bias = tf.get_variable('bias', [C], initializer=tf.constant_initializer(0.0))

    scale = tf.reshape(scale, [1, C, 1, 1])
    bias = tf.reshape(bias, [1, C, 1, 1])

    output = x * scale + bias
    output = tf.transpose(output, [0, 2 , 3, 1])

    return output