from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
keras = tf.keras

def maxpool(x, dim=-1, keepdim=False):
  out = tf.reduce_max(x, axis = dim, keep_dims = keepdim)

class SimplePointNet(keras.Model):

  def __init__(self, feature_dims, hidden_dim = 128):
    super(SimplePointNet, self).__init__()
    self.fcp = keras.layers.Dense(2*hidden_dim, activation = None)
    self.fc0 = keras.layers.Dense(hidden_dim, activation = None)
    self.fc1 = keras.layers.Dense(hidden_dim, activation = None)
    self.fc2 = keras.layers.Dense(hidden_dim, activation = None)
    self.fc3 = keras.layers.Dense(hidden_dim, activation = None)
    self.fch = keras.layers.Dense(feature_dims, activation = None)

    self.relu = keras.layers.ReLU()
    self.pool = maxpool
    layers = [2, 2, 2, 2]

  def call(self, x, training=False):
    # input size: (B, N, D)
    n_point = tf.shape(x)[1]
    x = self.fcp(x)
    x = self.fc0(self.relu(x))
    # pooled = self.pool(x, dim = 1, keepdim = True)
    pooled = tf.tile(tf.reduce_max(x, axis = 1, keepdims = True), [1, n_point, 1])
    x = tf.concat([x, pooled], axis = -1)

    x = self.fc1(self.relu(x))
    # pooled = self.pool(x, dim = 1, keepdim = True)
    pooled = tf.tile(tf.reduce_max(x, axis = 1, keepdims = True), [1, n_point, 1])
    x = tf.concat([x, pooled], axis = -1)
    
    x = self.fc2(self.relu(x))
    # pooled = self.pool(x, dim = 1, keepdim = True)
    pooled = tf.tile(tf.reduce_max(x, axis = 1, keepdims = True), [1, n_point, 1])
    x = tf.concat([x, pooled], axis = -1)

    x = self.fc3(self.relu(x))
    # x = self.pool(x, dim = 1)
    x = tf.reduce_max(x, axis = 1)

    x = self.fch(self.relu(x))

    return x

