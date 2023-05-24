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
from lib import convexdecoder

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
    self._dims = 3

    self._init_lr = args.lr
    self._batch_size = args.batch_size

    # self._useS = args.use_surface_sampling

    # Params = Roundness + Translation + Hyperplanes
    self.n_params = self._n_parts * (self._dims * self._n_vertices) + self._n_parts

    with tf.variable_scope("mc_autoencoder"):
      self.point_encoder = pointnet.SimplePointNet(args.latent_size)
      self.beta_decoder = Decoder(self.n_params)

    with tf.variable_scope("mc_convex"):
      self.cvx = convexdecoder.ConvexDecoder(args.n_parts, args.n_convex_altitude)

  def compute_loss(self, batch, training, optimizer=None):
    """Compute loss given a batch of data.

    Args:
      batch: Dict, must contains:
        "point": [batch_size, sample_size, 3],
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

    out_points, direction_h, overlap, distance, surf_distance, trans, vertices, smoothness, iter, iter_dist, undef, undef_dist = self.decode(
      beta, points, training=training)

    # h_loss = self._compute_h_loss(points, direction_h, trans)
    dist_loss = self._compute_distance_loss(distance)
    # s_loss = self._compute_sample_loss(points, out_points)
    overlap_loss = self._compute_overlap_loss(overlap)
    in_loss = self._compute_inside_loss(surf_distance, out_points, points)

    loss = dist_loss + 0.1*overlap_loss + in_loss
    # loss = h_loss + s_loss + 0.1*overlap_loss
    # loss = h_loss + s_loss

    if training:
      tf.summary.scalar("loss", loss)
      # tf.summary.scalar("x", tf.shape(vertices)[0])
      # tf.summary.scalar("y", tf.shape(vertices)[1])
      # tf.summary.scalar("z", tf.shape(vertices)[2])
      tf.summary.scalar("p1", smoothness[0, 0])
      # tf.summary.scalar("p2", smoothness[0, 1])
      # tf.summary.scalar("x", out_points[0, 0, 0])
      # tf.summary.scalar("y", out_points[0, 0, 1])
      # tf.summary.scalar("z", out_points[0, 0, 2])

    if training:
      global_step = tf.train.get_or_create_global_step()
      update_ops = self.updates
      lr = tf.train.exponential_decay(self._init_lr, global_step, 10000, 0.95, staircase = True)
      optimizer = optimizer(lr)
      # optimizer = optimizer(self._init_lr)

      with tf.control_dependencies(update_ops):
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients = [tf.where(tf.is_finite(grad) == False, tf.zeros_like(grad), grad) for grad in gradients]
        gradients, unused_var = tf.clip_by_global_norm(gradients, 1.0)
        train_op = optimizer.apply_gradients(
            zip(gradients, variables), global_step=global_step)
      return loss, train_op, global_step, out_points, vertices, smoothness, points, overlap, distance, iter, iter_dist, undef, undef_dist
    # else:
    #   occ = tf.cast(output >= self._level_set, tf.float32)
    #   intersection = tf.reduce_sum(occ * gt, axis=(1, 2))
    #   union = tf.reduce_sum(tf.maximum(occ, gt), axis=(1, 2))
    #   iou = intersection / union
    #   return loss, iou

  def encode(self, points, training):
    """Encode the input points into support function parameters.

    Args:
      points: Tensor, [batch_size, sample_size, 3]
      training: bool, use training mode if true.

    Returns:
      params: Tensor, [batch_size, n_params], support function parameters.
    """
    x = self.point_encoder(points, training=training)
    return self.beta_decoder(x, training=training)

  def decode(self, params, pointcloud, training):
    """Decode the support function parameters into indicator fuctions.

    Args:
      params: Tensor, [batch_size, n_params], hyperplane parameters.
      training: bool, use training mode if true.

    Returns:
      out_points: Tensor, [batch_size, n_out_points, 3], output surface point samples.
    """
    vertices, smoothness = self._split_params(params)
    out_points, direction_h, overlap, distance, surf_distance, trans, iter, iter_dist, undef, undef_dist = self.cvx(vertices, smoothness, pointcloud)
    return out_points, direction_h, overlap, distance, surf_distance, trans, vertices, smoothness, iter, iter_dist, undef, undef_dist

  def _split_params(self, params):
    """Split the parameter tensor."""
    vertices, smoothness = tf.split(
        params, [
            self._dims*self._n_parts*self._n_vertices, self._n_parts
        ],
        axis=-1)
    vertices = tf.reshape(vertices, [-1, self._n_parts, self._n_vertices, self._dims])
    smoothness = tf.reshape(smoothness, [-1, self._n_parts])
    vertices, smoothness = self._clamp_params(vertices, smoothness, 1, 40)
    return vertices, smoothness

  def _clamp_params(self, vertices, smoothness, vert_mag, max_p):
    vert = tf.tanh(vertices * vert_mag) / 2
    # vertices = tf.sigmoid(50*vertices) - 0.5
    # mask = tf.ones(tf.shape(p)) * 1e-20
    # p = tf.where(tf.is_nan(p), mask, p)
    p = (tf.tanh(smoothness) + 1) * max_p / 2 + 2
    # p = tf.nn.sigmoid(smoothness) * max_p + 1
    return vert, p

  def _compute_sample_loss(self, gt, output):
    gt = tf.expand_dims(gt, axis = 1)
    output = tf.expand_dims(output, axis = 2)
    directions = output - gt
    x = directions[:, :, :, 0]
    y = directions[:, :, :, 1]
    z = directions[:, :, :, 2]
    distances = x*x + y*y + z*z
    sample_loss = tf.reduce_min(distances, axis = 2)
    sample_loss = tf.reduce_mean(sample_loss)
    return sample_loss

  def _compute_h_loss(self, gt, output, translations):
    gt_points = tf.expand_dims(gt, axis = 2)
    gt_points = tf.tile(gt_points, [1, 1, tf.shape(translations)[1], 1])
    trans = tf.expand_dims(translations, axis = 1)
    trans = tf.tile(trans, [1, tf.shape(gt_points)[1], 1, 1])

    gt_local = gt_points - trans
    gt_local = tf.transpose(gt_local, [0, 2, 1, 3])

    out_points = output[:, :, :, 0:3]
    out_points = tf.transpose(out_points, [0, 1, 3, 2])

    dot = tf.matmul(gt_local, out_points)
    dot = tf.transpose(dot, [0, 2, 1, 3])

    h = output[:, :, :, 3]
    h = tf.expand_dims(h, axis = 1)
    h = tf.tile(h, [1, tf.shape(dot)[1], 1, 1])

    outsurf = dot - h
    outsurf = outsurf * tf.cast(outsurf > 0, tf.float32)
    # outsurf = tf.clip_by_value(outsurf, clip_value_min = 1e-10, clip_value_max = 1e+10) - tf.cast(outsurf < 1e-10, tf.float32)*1e-10
    outsurf = outsurf*outsurf
    outsurf = tf.reduce_max(outsurf, axis = -1)
    outsurf = tf.reduce_min(outsurf, axis = -1)

    insurf = h - dot

    # isin = tf.cast(insurf >= 0, tf.float32)
    # isin = tf.reduce_min(isin, axis = -1)

    # insurf = insurf * tf.cast(insurf >= 0, tf.float32) + tf.cast(insurf < 0, tf.float32)*10
    # # insurf = tf.clip_by_value(insurf, clip_value_min = 1e-10, clip_value_max = 1e+10) - tf.cast(insurf < 1e-10, tf.float32)*1e-10
    # insurf = insurf*insurf
    # insurf = tf.reduce_min(insurf, axis = -1)

    # insurf = insurf * isin
    insurf = insurf * tf.cast(insurf >= 0, tf.float32)
    insurf = insurf*insurf
    insurf = tf.reduce_min(insurf, axis = -1)
    insurf = tf.reduce_max(insurf, axis = -1)

    h_loss = tf.reduce_mean(outsurf + 10*insurf)
    return h_loss

  def _compute_overlap_loss(self, overlap):
    overlap_loss = overlap - 1.0
    overlap_loss = tf.reduce_mean(overlap_loss * overlap_loss)
    return overlap_loss
  
  def _compute_distance_loss(self, distance):
    min_dist = tf.reduce_min(tf.abs(distance), axis = 1)
    dist_loss = min_dist
    dist_loss = tf.reduce_mean(dist_loss)

    # in_point = -distance * tf.cast(distance < 0, tf.float32)
    # in_loss = tf.reduce_mean(in_point)
    return dist_loss
  
  def _compute_inside_loss(self, surf_distance, surf_points, points):
    # directions = tf.constant([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]])
    # directions = tf.cast(directions, tf.float32)
    # directions = directions / tf.linalg.norm(directions, axis = -1, keepdims = True)
    
    # directions = tf.tile(tf.reshape(directions, [1, 1, 14, 3]), [tf.shape(vertices)[0], tf.shape(vertices)[1], 1, 1]) # (B,C,14,3)
    # directions = tf.tile(tf.expand_dims(directions, axis = 2), [1, 1, tf.shape(points)[1], 1, 1]) # (B,C,P,14,3)
    # directions = tf.transpose(directions, [0, 1, 2, 4, 3]) # (B,C,P,3,14)

    # center = tf.reduce_mean(vertices, axis = 2, keepdims = True) # (B,C,1,3)
    # local_points = tf.expand_dims(points, axis = 1) - center # (B,C,P,3)
    # local_points = local_points / tf.sqrt(tf.reduce_sum(local_points*local_points, axis = -1, keepdims = True) + 1e-30)
    # local_points = tf.expand_dims(local_points, axis = 3) # (B,C,P,1,3)

    # in_loss = tf.squeeze(tf.matmul(local_points, directions), axis = -2) # (B,C,P,14)
    # in_loss = tf.reduce_max(in_loss, axis = -2) # (B,C,14)
    # in_loss = 1 - in_loss
    # in_loss = tf.reduce_mean(in_loss)

    point_dist = tf.expand_dims(points, axis = 2) - tf.expand_dims(surf_points, axis = 1) # (B,P,CD,3)
    point_dist = tf.reduce_sum(point_dist*point_dist, axis = -1, keepdims = True) # (B,P,CD,1)
    in_loss = tf.concat([surf_distance*surf_distance, point_dist], axis = 1) # (B,C+P,CD,1)
    in_loss = tf.reduce_min(in_loss, axis = 1) # (B,CD,1)
    in_loss = tf.reduce_mean(in_loss)
    
    return in_loss



class Decoder(keras.layers.Layer):
  """MLP decoder to decode latent codes to support function parameters."""

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
