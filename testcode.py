from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow as tf2
import os
from pyntcloud import PyntCloud
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

print(np.random.rand(3, ))