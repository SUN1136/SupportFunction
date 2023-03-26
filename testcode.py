from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
keras = tf.keras

# x = keras.Sequential()
# x.add(keras.layers.Dense(10, input_shape = (None, 1024, 3)))
# x.summary()

# a = tf.constant([1, 2, 3, 4])
# a = tf.reshape(a, [2, 2])
# print(1/a)

x = tf.constant([[[1, 2, 3, 4], [5, 6, 7, 8]], [[1, 2, 45, 1], [12,2, 5, 3]]])
z, idx = tf.nn.top_k(x, k = 2)
print(z, idx)
