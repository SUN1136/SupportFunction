# Copyright 2020 The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model Implementations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from lib import pointnet

keras = tf.keras


def get_model(model_name, args):
  return model_dict[model_name](args)


class MultiConvexNet(keras.Model):
  """Shape auto-encoder with multiple convex polytopes.

  Attributes:
    n_params: int, the number of hyperplane parameters.
  """

  def __init__(self, args):
    super(MultiConvexNet, self).__init__()
    self._n_parts = args.n_parts
    self._n_vertices = args.n_vertices
    self._sample_size = args.sample_point
    self._image_input = args.image_input
    self._dims = args.dims

    self._batch_size = args.batch_size

    # Params = Roundness + Translation + Hyperplanes
    self.n_params = self._n_parts * (self._dims * self._n_vertices) + self._n_parts

    with tf.variable_scope("mc_autoencoder"):
      self.point_encoder = pointnet.SimplePointNet(args.latent_size)
      self.beta_decoder = Decoder(self.n_params)

    with tf.variable_scope("mc_convex"):
      self.cvx = ConvexSurfaceSampler(args.dims, args.n_parts, args.n_convex_altitude)

  def compute_loss(self, batch, training, optimizer=None):
    """Compute loss given a batch of data.

    Args:
      batch: Dict, must contains:
        "point": [batch_size, sample_size, dims],
      training: bool, use training mode if true.
      optimizer: tf.train.Optimizer, optimizer used for training.

    Returns:
      train_loss: tf.Operation, loss hook.
      train_op: tf.Operation, optimization op.
      global_step: tf.Operation, gloabl step hook.
    """
    points = batch["point"]

    if not self._image_input:
      beta = self.encode(points, training=training)
    
    out_points, vertices, smoothness, direc, locvert, dhdz, zm = self.decode(beta, training=training)

    out2in = self._compute_sample_loss(points, out_points)
    in2out = self._compute_sample_loss(out_points, points)
    loss = out2in + in2out

    if training:
      tf.summary.scalar("loss", loss)
      # tf.summary.scalar("x", tf.shape(vertices)[0])
      # tf.summary.scalar("y", tf.shape(vertices)[1])
      # tf.summary.scalar("z", tf.shape(vertices)[2])
      tf.summary.scalar("x", out_points[0, 0, 0])
      tf.summary.scalar("y", out_points[0, 0, 1])
      tf.summary.scalar("z", out_points[0, 0, 2])

    if training:
      global_step = tf.train.get_or_create_global_step()
      update_ops = self.updates
      with tf.control_dependencies(update_ops):
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, unused_var = tf.clip_by_global_norm(gradients, 5.0)
        train_op = optimizer.apply_gradients(
            zip(gradients, variables), global_step=global_step)
      return loss, train_op, global_step, out_points, beta, vertices, smoothness, direc, locvert, dhdz, zm
    # else:
    #   occ = tf.cast(output >= self._level_set, tf.float32)
    #   intersection = tf.reduce_sum(occ * gt, axis=(1, 2))
    #   union = tf.reduce_sum(tf.maximum(occ, gt), axis=(1, 2))
    #   iou = intersection / union
    #   return loss, iou

  def encode(self, points, training):
    """Encode the input points into support function parameters.

    Args:
      points: Tensor, [batch_size, sample_size, dims]
      training: bool, use training mode if true.

    Returns:
      params: Tensor, [batch_size, n_params], support function parameters.
    """
    x = self.point_encoder(points, training=training)
    return self.beta_decoder(x, training=training)

  def decode(self, params, training):
    """Decode the support function parameters into indicator fuctions.

    Args:
      params: Tensor, [batch_size, n_params], hyperplane parameters.
      training: bool, use training mode if true.

    Returns:
      out_points: Tensor, [batch_size, n_out_points, dims], output surface point samples.
    """
    vertices, smoothness = self._split_params(params)
    out_points, direc, locvert, dhdz, zm = self.cvx(vertices, smoothness)
    return out_points, vertices, smoothness, direc, locvert, dhdz, zm

  def _split_params(self, params):
    """Split the parameter tensor."""
    vertices, smoothness = tf.split(
        params, [
            self._dims*self._n_parts*self._n_vertices, self._n_parts
        ],
        axis=-1)
    vertices = tf.reshape(vertices, [-1, self._n_parts, self._n_vertices, self._dims])
    smoothness = tf.reshape(smoothness, [-1, self._n_parts])
    vertices, smoothness = self._clamp_params(vertices, smoothness, 1, 60)
    return vertices, smoothness
  
  def _clamp_params(self, vertices, smoothness, vert_mag, max_p):
    vert = tf.tanh(vertices * vert_mag) / 2
    # vertices = tf.sigmoid(50*vertices) - 0.5
    p = (tf.tanh(smoothness) + 2) * max_p / 3
    # p = tf.nn.sigmoid(smoothness) * 10 + 1
    return vert, p

  def _compute_sample_loss(self, gt, output):
    gt = tf.expand_dims(gt, axis = 1)
    output = tf.expand_dims(output, axis = 2)
    directions = output - gt
    x = directions[:, :, :, 0]
    y = directions[:, :, :, 1]
    z = directions[:, :, :, 2]
    distances = tf.pow(x, 2) + tf.pow(y, 2) + tf.pow(z, 2)
    sample_loss = tf.reduce_min(distances, axis = 2)
    sample_loss = tf.reduce_mean(sample_loss)
    return sample_loss




class ConvexSurfaceSampler(keras.layers.Layer):
  """Differentiable shape rendering layer using multiple convex polytopes."""

  def __init__(self, dims, n_parts, n_th1):
    super(ConvexSurfaceSampler, self).__init__()

    self._dims = dims
    self._n_parts = n_parts
    self._th1_range = np.pi / 2
    self._th2_range = np.pi

    self._n_th1 = n_th1
    self._n_th2 = 2*(n_th1 - 1) + 1

    self._n_th = self._n_th1 * self._n_th2
    self._n_out_points = (self._n_th1 - 2) * (self._n_th2 - 1) + 2 
    self._n_mesh = (self._n_th1 - 1) * (self._n_th2 - 1)

    self._lb = 1e-20
    self._ub = 1e+20
    self._lb_exponent = -20
    self._ub_exponent = 20
  

  def call(self, vertices, smoothness):
    """Decode the support function parameters into surface point samples.

    Args:
      vertices: Tensor, [batch_size, n_parts, n_vertices, dims], convex vertices.
      smoothness: Tensor, [batch_size, n_parts], convex smoothness.

    Returns:
      points: Tensor, [batch_size, n_surface_points, dims], output surface point samples.
    """
    th1 = tf.linspace(-self._th1_range, self._th1_range, self._n_th1)
    th2 = tf.linspace(-self._th2_range, self._th2_range, self._n_th2)

    cos_th1 = tf.cos(th1)
    sin_th1 = tf.sin(th1)
    cos_th2 = tf.cos(th2)
    sin_th2 = tf.sin(th2)

    # if self._th1_range == np.pi/2:
    #   cos_th1[0] = 0
    #   cos_th1[-1] = 0
    #   sin_th1[0] = -1
    #   sin_th1[-1] = 1
    # if self._th2_range == np.pi:
    #   cos_th2[0] = -1
    #   cos_th2[-1] = -1
    #   sin_th2[0] = 0
    #   sin_th2[-1] = 0
    # if self._n_th1 % 2 != 0:
    #   mid = (self._n_th1 - 1) / 2
    #   cos_th1[mid] = 1
    #   sin_th1[mid] = 0
    #   cos_th2[mid] = 1
    #   sin_th2[mid] = 0
    
    directions = []
    for i in range(1, self._n_th1 - 1):
      for j in range(self._n_th2 - 1):
        idx = (self._n_th2 - 1)*(i - 1) + j
        d_x = cos_th1[i]*cos_th2[j]
        d_y = cos_th1[i]*sin_th2[j]
        d_z = sin_th1[i]
        d = tf.reshape(tf.concat([[d_x], [d_y], [d_z]], axis = 0), [3, ])
        directions.append(d)
    directions.append(tf.reshape(tf.concat([[cos_th1[0]*cos_th2[0]], [cos_th1[0]*sin_th2[0]], [sin_th1[0]]], axis = 0), [3, ]))
    directions.append(tf.reshape(tf.concat([[cos_th1[-1]*cos_th2[0]], [cos_th1[-1]*sin_th2[0]], [sin_th1[-1]]], axis = 0), [3, ]))
    directions = tf.concat(directions, axis = 0)
    directions = tf.reshape(directions, [-1, self._dims])

    """
    mean_vertices: Tensor, [batch, n_parts, 1, dims], centor of geometry for each convex
    local_vertices: Tensor, [batch, n_parts, n_vertices, dims], vertices on convex local coordinate
    """
    mean_vertices = tf.reduce_mean(vertices, axis = 2, keepdims = True)
    local_vertices = vertices - mean_vertices

    points, dhdz, zm = self._compute_spt(directions, local_vertices, smoothness, mean_vertices)

    return points, directions, local_vertices, dhdz, zm

  def _compute_spt(self, x, vertices, smoothness, translations):
    """Compute support function and surface point samples.

    Args:
      x: Tensor, [n_th, dims], direction samples.
      vertices: Tensor, [batch_size, n_parts, n_vertices, dims], convex local vertices.
      smoothness: Tensor, [batch_size, n_parts], convex smoothness.
      translations: Tensor, [batch_size, n_parts, 1, dims], convex centers.

    Returns:
      points: Tensor, [batch_size, n_surface_points, dims], output surface point samples.
    """
    x = tf.transpose(x, [1, 0])
    x = tf.expand_dims(tf.expand_dims(x, axis = 0), axis = 0)
    x = tf.tile(x, [tf.shape(vertices)[0], tf.shape(vertices)[1], 1, 1])

    "z: Tensor, [batch_size, n_parts, n_vertices, n_out_points], dot product of v and x"
    z = tf.matmul(vertices, x)
    zm = tf.cast(z > 0, tf.float32) * z

    n_v = tf.shape(zm)[2]
    n_d = tf.shape(zm)[3]

    p = tf.expand_dims(tf.expand_dims(smoothness, axis = -1), axis = -1)
    p1 = tf.tile(p, [1, 1, n_v, n_d])
    p2 = tf.tile(p, [1, 1, 1, n_d])

    zm_log = tf.log(zm) / tf.log(10.0) # usually, =< 0
    zm_log = tf.reduce_max(zm_log, axis = 2, keepdims = True)
    exponent = zm_log * p2
    # uk = tf.cast(exponent > self._ub_exponent, tf.float32) * ((self._ub_exponent - exponent) / p2)
    lk = tf.cast(exponent < self._lb_exponent, tf.float32) * ((self._lb_exponent - exponent) / p2)
    k = tf.clip_by_value(tf.ceil(lk), clip_value_min = 0, clip_value_max = self._ub)
    base = tf.ones(tf.shape(k)) * 10.0
    k = tf.pow(base, k)
    k = tf.tile(k, [1, 1, n_v, 1])

    zm = zm * k

    zm_p = tf.clip_by_value(tf.pow(zm, p1), clip_value_min = self._lb, clip_value_max = self._ub) - tf.cast(zm == 0, tf.float32) * self._lb
    sum_zm_p = tf.reduce_sum(zm_p, axis = 2, keepdims = True)
    # sum_zm = tf.clip_by_value(tf.pow(sum_zm_p, 1 / p2 - 1), clip_value_min = 0, clip_value_max = self._ub)
    # sum_zm = tf.pow(sum_zm_p, 1 / p2 - 1)
    # sum_zm = tf.tile(sum_zm, [1, 1, n_v, 1])
    sum_zm = tf.pow(sum_zm_p, 1 / p2)
    sum_zm = tf.tile(sum_zm, [1, 1, n_v, 1])

    # dhdz = sum_zm * zm_p_1
    dhdz = tf.pow((zm / sum_zm), (p1 - 1))
    dhdz_t = tf.transpose(dhdz, [0, 1, 3, 2])
    dhdx = tf.matmul(dhdz_t, vertices)

    points = dhdx + translations
    points = self._remove_overlap(points)

    return points, dhdz, zm

  def _remove_overlap(self, x):
    n_b = tf.shape(x)[0]
    n_dims = tf.shape(x)[-1]
    points = tf.reshape(x, [n_b, -1, n_dims])
    return points


class Decoder(keras.layers.Layer):
  """MLP decoder to decode latent codes to hyperplane parameters."""

  def __init__(self, dims):
    super(Decoder, self).__init__()
    self._decoder = keras.Sequential()
    layer_sizes = [1024, 1024, 1024]
    for layer_size in layer_sizes:
      self._decoder.add(
          keras.layers.Dense(layer_size, activation=tf.nn.leaky_relu))
    self._decoder.add(keras.layers.Dense(dims, activation=None))

  def call(self, x, training):
    return self._decoder(x, training=training)


model_dict = {
    "multiconvex": MultiConvexNet,
}
