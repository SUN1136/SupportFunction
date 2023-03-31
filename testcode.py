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

x = tf.constant([[1, 2], [-4, 1]])
y = tf.pow(10.0, -40)

print(tf.shape(x), y)
