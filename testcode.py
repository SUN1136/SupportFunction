from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow as tf2
import os
keras = tf.keras

# x = keras.Sequential()
# x.add(keras.layers.Dense(10, input_shape = (None, 1024, 3)))
# x.summary()

# a = tf.constant([1, 2, 3, 4])
# a = tf.reshape(a, [2, 2])
# print(1/a)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# gpus = tf2.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # memory limit 10 times increased
#         tf2.config.experimental.set_virtual_device_configuration(gpus[0], [tf2.config.experimental.VirtualDeviceConfiguration(memory_limit=100000)])
#         # tf.config.experimental.set_memory_growth(gpus[0], True)
#         print("\n----------GPU Loaded----------\n")
#     except RuntimeError as e:
#         print(e)

a = tf.constant([1.5, 20.2, 61.2, 1.513, 6.124, 0.0021, 0.515, 1.2385, 0.121, 3.22, 0.000001, 51.22 ,2e-20, 3.124e-6, 2.121e-12, 1.112e-4])
a = tf.reshape(a, [4, 4])
print(a)
print(tf.linalg.inv(a))
print(tf.linalg.pinv(a))
print("")
maxa = tf.reduce_max(a)
a = a / maxa
print(a)
print(tf.linalg.inv(a) / maxa)
print(tf.linalg.pinv(a) / maxa)
