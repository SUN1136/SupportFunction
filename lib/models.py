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
import sys

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
      self.cvx = ConvexSurfaceSampler(args.n_parts, args.n_convex_altitude)

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

    out_points, direction_h, overlap, trans, vertices, smoothness, direc, locvert, dhdz, zm, iter, undef = self.decode(beta, training=training)

    # if self._useS:
    #   out2in = self._compute_sample_loss(points, out_points)
    #   in2out = self._compute_sample_loss(out_points, points)
    #   loss = out2in + in2out
    # else:
    h_loss = self._compute_h_loss(points, direction_h, trans)
    s_loss = self._compute_sample_loss(points, out_points)
    overlap_loss = self._compute_overlap_loss(overlap)

    loss = h_loss + s_loss + 0.1*overlap_loss
    # loss = s_loss + 0.1*overlap_loss

    if training:
      tf.summary.scalar("loss", loss)
      # tf.summary.scalar("x", tf.shape(vertices)[0])
      # tf.summary.scalar("y", tf.shape(vertices)[1])
      # tf.summary.scalar("z", tf.shape(vertices)[2])
      tf.summary.scalar("p1", smoothness[0, 0])
      # tf.summary.scalar("p2", smoothness[0, 1])
      tf.summary.scalar("x", out_points[0, 0, 0])
      tf.summary.scalar("y", out_points[0, 0, 1])
      tf.summary.scalar("z", out_points[0, 0, 2])

    if training:
      global_step = tf.train.get_or_create_global_step()
      update_ops = self.updates
      lr = tf.train.exponential_decay(self._init_lr, global_step, 10000, 0.95, staircase = True)
      optimizer = optimizer(lr)

      with tf.control_dependencies(update_ops):
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients = [tf.where(tf.is_finite(grad) == False, tf.zeros_like(grad), grad) for grad in gradients]
        gradients, unused_var = tf.clip_by_global_norm(gradients, 1.0)
        train_op = optimizer.apply_gradients(
            zip(gradients, variables), global_step=global_step)
      return loss, train_op, global_step, out_points, beta, vertices, smoothness, direc, locvert, dhdz, zm, points, overlap, iter, undef
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

  def decode(self, params, training):
    """Decode the support function parameters into indicator fuctions.

    Args:
      params: Tensor, [batch_size, n_params], hyperplane parameters.
      training: bool, use training mode if true.

    Returns:
      out_points: Tensor, [batch_size, n_out_points, 3], output surface point samples.
    """
    vertices, smoothness = self._split_params(params)
    out_points, direction_h, overlap, trans, direc, locvert, dhdz, zm, iter, undef = self.cvx(vertices, smoothness)
    return out_points, direction_h, overlap, trans, vertices, smoothness, direc, locvert, dhdz, zm, iter, undef

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

    isin = tf.cast(insurf >= 0, tf.float32)
    isin = tf.reduce_min(isin, axis = -1)

    insurf = insurf * tf.cast(insurf >= 0, tf.float32) + tf.cast(insurf < 0, tf.float32)*10
    # insurf = tf.clip_by_value(insurf, clip_value_min = 1e-10, clip_value_max = 1e+10) - tf.cast(insurf < 1e-10, tf.float32)*1e-10
    insurf = insurf*insurf
    insurf = tf.reduce_min(insurf, axis = -1)

    insurf = insurf * isin
    insurf = tf.reduce_max(insurf, axis = -1)

    h_loss = tf.reduce_mean(outsurf + 10*insurf)
    return h_loss

  def _compute_overlap_loss(self, overlap):
    overlap_loss = overlap - 1.0
    overlap_loss = tf.reduce_mean(overlap_loss * overlap_loss)
    return overlap_loss




class ConvexSurfaceSampler(keras.layers.Layer):
  """Differentiable shape rendering layer using multiple convex polytopes."""

  def __init__(self, n_parts, n_th1):
    super(ConvexSurfaceSampler, self).__init__()

    # self._useS = useS

    self._dims = 3
    self._n_parts = n_parts
    self._th1_range = np.pi / 2
    self._th2_range = np.pi
    # self._n_mesh_inter = n_mesh_inter

    self._n_th1 = n_th1
    self._n_th2 = 2*(n_th1 - 1) + 1

    self._n_th = self._n_th1 * self._n_th2
    self._n_out_points = (self._n_th1 - 2) * (self._n_th2 - 1) + 2
    self._n_mesh = (self._n_th1 - 2)*(self._n_th2 - 1)*2

    self._lb = 1e-20
    self._ub = 1e+20
    self._lb_exponent = -20
    self._ub_exponent = 20

    th1 = tf.linspace(-self._th1_range, self._th1_range, self._n_th1)
    th2 = tf.linspace(-self._th2_range, self._th2_range, self._n_th2)
    cos_th1 = tf.cos(th1)
    sin_th1 = tf.sin(th1)
    cos_th2 = tf.cos(th2)
    sin_th2 = tf.sin(th2)

    self.directions = []
    for i in range(1, self._n_th1 - 1):
      for j in range(self._n_th2 - 1):
        # idx = (self._n_th2 - 1)*(i - 1) + j
        d_x = cos_th1[i]*cos_th2[j]
        d_y = cos_th1[i]*sin_th2[j]
        d_z = sin_th1[i]
        d = tf.reshape(tf.concat([[d_x], [d_y], [d_z]], axis = 0), [3, ])
        self.directions.append(d)
    self.directions.append(tf.reshape(tf.concat([[cos_th1[0]*cos_th2[0]], [cos_th1[0]*sin_th2[0]], [sin_th1[0]]], axis = 0), [3, ]))
    self.directions.append(tf.reshape(tf.concat([[cos_th1[-1]*cos_th2[0]], [cos_th1[-1]*sin_th2[0]], [sin_th1[-1]]], axis = 0), [3, ]))
    self.directions = tf.concat(self.directions, axis = 0)
    self.directions = tf.reshape(self.directions, [-1, self._dims])

    # self.mesh_idx = []
    # for i in range(1, self._n_th1 - 2):
    #   for j in range(self._n_th2 - 1):
    #     first = (i - 1)*(self._n_th2 - 1) + j
    #     second = (i - 1)*(self._n_th2 - 1) + j + 1 if j < self._n_th2 - 2 else (i - 1)*(self._n_th2 - 1)
    #     third = i*(self._n_th2 - 1) + j
    #     fourth = i*(self._n_th2 - 1) + j + 1 if j < self._n_th2 - 2 else i*(self._n_th2 - 1)
    #     idx = tf.reshape(tf.constant([first, second, third], dtype = tf.int32), [1, 3])
    #     self.mesh_idx.append(idx)
    #     idx = tf.reshape(tf.constant([second, third, fourth], dtype = tf.int32), [1, 3])
    #     self.mesh_idx.append(idx)
    # for j in range(self._n_th2 - 1):
    #   first = (self._n_th1 - 2)*(self._n_th2 - 1)
    #   second = j
    #   third = j + 1 if j < self._n_th2 - 2 else 0
    #   idx = tf.reshape(tf.constant([first, second, third], dtype = tf.int32), [1, 3])
    #   self.mesh_idx.append(idx)

    #   first = (self._n_th1 - 2)*(self._n_th2 - 1) + 1
    #   second = (self._n_th1 - 3)*(self._n_th2 - 1) + j
    #   third = (self._n_th1 - 3)*(self._n_th2 - 1) + j + 1 if j < self._n_th2 - 2 else (self._n_th1 - 3)*(self._n_th2 - 1)
    #   idx = tf.reshape(tf.constant([first, second, third], dtype = tf.int32), [1, 3])
    #   self.mesh_idx.append(idx)
    # self.mesh_idx = tf.concat(self.mesh_idx, axis = 0)

    # self.ntop = ntop
    # self.nbottom = nbottom
    # self.mesh_filter = []
    # for i in range(self._n_mesh):
    #   tmp = tf.ones([1, 1, 1]) * i
    #   self.mesh_filter.append(tmp)
    # self.mesh_filter = tf.concat(self.mesh_filter, axis = -1)
    # self.mesh_filter = tf.expand_dims(self.mesh_filter, axis = -1)


  def call(self, vertices, smoothness):
    """Decode the support function parameters into surface point samples.

    Args:
      vertices: Tensor, [batch_size, n_parts, n_vertices, 3], convex vertices.
      smoothness: Tensor, [batch_size, n_parts], convex smoothness.

    Returns:
      points: Tensor, [batch_size, n_surface_points, 3], output surface point samples.
    """

    """
    mean_vertices: Tensor, [batch, n_parts, 1, 3], centor of geometry for each convex
    local_vertices: Tensor, [batch, n_parts, n_vertices, 3], vertices on convex local coordinate
    """
    mean_vertices = tf.reduce_mean(vertices, axis = 2, keepdims = True)
    local_vertices = vertices - mean_vertices

    # if self._useS:
    #   points, dhdz, zm = self._compute_spt(self.directions, self.mesh_idx, local_vertices, smoothness, mean_vertices)
    #   direction_h = tf.zeros([1, 1, 1, 1])
    # else:
    direction_h, points, overlap, iter, undef = self._compute_output(self.directions, local_vertices, smoothness, mean_vertices)
    mean_vertices = tf.reshape(mean_vertices, [tf.shape(mean_vertices)[0], tf.shape(mean_vertices)[1], tf.shape(mean_vertices)[-1]])
    dhdz = tf.zeros([1, 1, 1, 1])
    zm = tf.zeros([1, 1, 1, 1])

    return points, direction_h, overlap, mean_vertices, self.directions, local_vertices, dhdz, zm, iter, undef

  def _compute_output(self, x, vertices, smoothness, translations):
    """Compute output.

    Args:
      x: Tensor, [n_direction, 3], direction samples.
      vertices: Tensor, [batch_size, n_parts, n_vertices, 3], convex local vertices.
      smoothness: Tensor, [batch_size, n_parts], convex smoothness.
      translations: Tensor, [batch_size, n_parts, 1, 3], convex centers.

    Returns:
      direction_h: Tensor, [batch_size, n_parts, n_directions, 3 + 1], global directions and h(x).
      surf_points: Tensor, [batch_size, n_surface_points, 3], output surface point samples.
    """
    local_directions = tf.expand_dims(tf.expand_dims(x, axis = 0), axis = 0)
    local_directions = tf.tile(local_directions, [tf.shape(vertices)[0], tf.shape(vertices)[1], 1, 1])

    x = tf.transpose(x, [0, 1])
    x = tf.expand_dims(tf.expand_dims(x, axis = 0), axis = 0)
    x = tf.tile(x, [tf.shape(vertices)[0], tf.shape(vertices)[1], 1, 1])

    h, surf_points = self._compute_spt(x, vertices, smoothness, translations)

    direction_h = tf.concat([local_directions, h], axis = -1)
    surf_points = tf.reshape(surf_points, [tf.shape(surf_points)[0], -1, tf.shape(surf_points)[-1]])

    overlap, iter, undef = self._compute_overlap(vertices, smoothness, translations)

    return direction_h, surf_points, overlap, iter, undef

  def _compute_spt(self, x, vertices, smoothness, translations, get_h = True, get_dsdx = False):
    """Compute output.

    Args:
      x: Tensor, [batch_size, n_parts, n_direction, 3], direction samples.
      vertices: Tensor, [batch_size, n_parts, n_vertices, 3], convex local vertices.
      smoothness: Tensor, [batch_size, n_parts], convex smoothness.
      translations: Tensor, [batch_size, n_parts, 1, 3], convex centers.
      get_dsdx: bool, whether return ds/dx or not

    Returns:
      h: Tensor, [batch_size, n_parts, n_directions, 1], output h(x).
      surf_points: Tensor, [batch_size, n_parts, n_directions, 3], output surface point samples.
      dsdx: Tensor, [batch_size, n_parts, n_directions, 3, 3], output ds/dx
    """

    "z: Tensor, [batch_size, n_parts, n_vertices, n_direction], dot product of v and x"
    x = tf.transpose(x, [0, 1, 3, 2])
    z = tf.matmul(vertices, x)
    zm = tf.cast(z > 0, tf.float32) * z

    n_v = tf.shape(zm)[2]
    n_d = tf.shape(zm)[3]

    p = tf.expand_dims(tf.expand_dims(smoothness, axis = -1), axis = -1)
    p1 = tf.tile(p, [1, 1, n_v, n_d])
    p2 = tf.tile(p, [1, 1, 1, n_d])

    zm_log = tf.log(zm) / tf.log(10.0)
    zm_log = tf.reduce_max(zm_log, axis = 2, keepdims = True)
    exponent = zm_log * p2
    lk = tf.cast(exponent < self._lb_exponent, tf.float32) * ((self._lb_exponent - exponent) / p2)
    k = tf.clip_by_value(tf.ceil(lk), clip_value_min = 0.0, clip_value_max = self._ub_exponent)
    base = tf.ones(tf.shape(k)) * 10.0
    k = tf.pow(base, k)
    k2 = k
    k = tf.tile(k, [1, 1, n_v, 1])

    zm = zm * k

    zm_p = tf.clip_by_value(tf.pow(zm, p1), clip_value_min = self._lb, clip_value_max = self._ub) - tf.cast(zm == 0, tf.float32) * self._lb
    sum_zm_p = tf.reduce_sum(zm_p, axis = 2, keepdims = True)
    h = tf.clip_by_value(tf.pow(sum_zm_p, 1 / p2), clip_value_min = self._lb, clip_value_max = self._ub)
    sum_zm = tf.tile(h, [1, 1, n_v, 1])


    "Compute dh/dx=s"
    dhdz = tf.clip_by_value(tf.pow((zm / sum_zm), (p1 - 1)), clip_value_min = self._lb, clip_value_max = self._ub)
    dhdz_t = tf.transpose(dhdz, [0, 1, 3, 2])
    dhdx = tf.matmul(dhdz_t, vertices)

    surf_points = dhdx + translations


    if get_dsdx:
      "Compute ds/dx"
      zm_p_2 = tf.clip_by_value(tf.pow(zm, p1 - 2), clip_value_min = self._lb, clip_value_max = self._ub)
      zm_p_2 = tf.transpose(zm_p_2, [0, 1, 3, 2])
      diag_zm_p_2 = tf.linalg.diag(zm_p_2)

      zm_p_1 = tf.clip_by_value(tf.pow(zm, p1 - 1), clip_value_min = self._lb, clip_value_max = self._ub)
      zm_p_1 = tf.transpose(zm_p_1, [0, 1, 3, 2])
      mat_zm_p_1 = tf.matmul(tf.expand_dims(zm_p_1, axis = -1), tf.expand_dims(zm_p_1, axis = -2))

      # dsdx = diag_zm_p_2 - tf.clip_by_value(mat_zm_p_1 / tf.expand_dims(tf.transpose(tf.clip_by_value(sum_zm_p, clip_value_min = self._lb, clip_value_max = self._ub), [0, 1, 3, 2]), axis = -1), clip_value_min = self._lb, clip_value_max = self._ub)

      sum_zm_1 = tf.clip_by_value(tf.pow(sum_zm_p, 1 / p2 - 1), clip_value_min = self._lb, clip_value_max = self._ub)
      sum_zm_1 = tf.expand_dims(tf.transpose(sum_zm_1, [0, 1, 3, 2]), axis = -1)
      sum_zm_2 = tf.clip_by_value(tf.pow(sum_zm_p, 1 / p2 - 2), clip_value_min = self._lb, clip_value_max = self._ub)
      sum_zm_2 = tf.expand_dims(tf.transpose(sum_zm_2, [0, 1, 3, 2]), axis = -1)

      v_left = tf.tile(tf.expand_dims(tf.transpose(vertices, [0, 1, 3, 2]), axis = 2), [1, 1, n_d, 1, 1])
      v_right = tf.tile(tf.expand_dims(vertices, axis = 2), [1, 1, n_d, 1, 1])

      dsdx = diag_zm_p_2*sum_zm_1 - mat_zm_p_1*sum_zm_2

      dsdx = tf.matmul(tf.matmul(v_left, dsdx), v_right)


      p3 = tf.expand_dims(tf.expand_dims(tf.expand_dims(smoothness, axis = -1), axis = -1), axis = -1)
      k3 = tf.expand_dims(tf.transpose(k2, [0, 1, 3, 2]), axis = -1)

      # dsdx = (p3 - 1.0) * k3 * sum_zm_1 * dsdx
      dsdx = (p3 - 1.0) * k3 * dsdx

      # dsdx_sgn = tf.sign(dsdx)
      # dsdx_sgn = dsdx_sgn + tf.cast(dsdx_sgn == 0.0, tf.float32)
      # dsdx = dsdx_sgn * tf.clip_by_value(tf.abs(dsdx), clip_value_min = self._lb, clip_value_max = self._ub)
      dsdx = tf.clip_by_value(dsdx, clip_value_min = -self._ub, clip_value_max = self._ub)

      if get_h:
        h = tf.clip_by_value(h / k2, clip_value_min = -self._ub, clip_value_max = self._ub)
        h = tf.transpose(h, [0, 1, 3, 2])
        return h, surf_points, dsdx
      else:
        return surf_points, dsdx

    else:
      if get_h:
        h = tf.clip_by_value(h / k2, clip_value_min = -self._ub, clip_value_max = self._ub)
        h = tf.transpose(h, [0, 1, 3, 2])
        return h, surf_points
      else:
        return surf_points

  def _compute_overlap(self, vertices, smoothness, translations):
    """Compute overlapping state.

    Args:
      vertices: Tensor, [batch_size, n_parts, n_vertices, 3], convex local vertices.
      smoothness: Tensor, [batch_size, n_parts], convex smoothness.
      translations: Tensor, [batch_size, n_parts, 1, 3], convex centers.

    Returns:
      overlap: Tensor, [batch_size, n_parts, 1], convex minimum growth rate.
    """
    n_c = self._n_parts

    overlap_list = []
    loop_iter = tf.constant([0])
    undef_list = []
    undef_j = tf.constant([0.0])
    undef = tf.constant([0.0])

    for i in range(n_c - 1):
      n_jc = n_c - 1 - i
      i_v = tf.tile(vertices[:, i:(i+1), :, :], [1, n_jc, 1, 1])
      i_p = tf.tile(smoothness[:, i:(i+1)], [1, n_jc])
      i_t = tf.tile(translations[:, i:(i+1), :, :], [1, n_jc, 1, 1])

      j_v = vertices[:, i+1:, :, :]
      j_p = smoothness[:, i+1:]
      j_t = translations[:, i+1:, :, :]

      o_bar = j_t - translations[:, i:(i+1), :, :]
      o_bar2 = tf.tile(tf.expand_dims(tf.transpose(o_bar, [0, 1, 3, 2]), axis = 2), [1, 1, 4, 1, 1])

      u_ie = o_bar / self._safe_norm(o_bar, axis = -1, keepdims = True)
      u0 = tf.tile(tf.reshape(tf.constant([1.0, 1.0, 1.0]), [1, 1, 3, 1]), [tf.shape(vertices)[0], n_jc, 1, 1])


      "IE Initialization"
      V_ie = 1e-3 * tf.constant([[1.0, 0.0, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5], [0.0, 0.0, 1.0]])
      V_ie = tf.tile(tf.reshape(tf.transpose(V_ie, [1, 0]), [1, 1, 3, 4]), [tf.shape(vertices)[0], n_jc, 1, 1])
      W_ie = tf.tile(tf.zeros([1, 1, 4, 3, 3]), [tf.shape(vertices)[0], n_c, 1, 1, 1])
      W_filter = tf.tile(tf.reshape(tf.constant([0.0, 1.0, 2.0, 3.0]), [1, 1, 4]), [tf.shape(vertices)[0], n_jc, 1])

      iter = 0
      for j in range(10):
        W_ie_0 = tf.expand_dims(tf.concat([V_ie[:, :, :, 1:2], V_ie[:, :, :, 2:3], V_ie[:, :, :, 3:4]], axis = -1), axis = 2)
        W_ie_1 = tf.expand_dims(tf.concat([V_ie[:, :, :, 0:1], V_ie[:, :, :, 2:3], V_ie[:, :, :, 3:4]], axis = -1), axis = 2)
        W_ie_2 = tf.expand_dims(tf.concat([V_ie[:, :, :, 0:1], V_ie[:, :, :, 1:2], V_ie[:, :, :, 3:4]], axis = -1), axis = 2)
        W_ie_3 = tf.expand_dims(tf.concat([V_ie[:, :, :, 0:1], V_ie[:, :, :, 1:2], V_ie[:, :, :, 2:3]], axis = -1), axis = 2)
        W_ie =  tf.concat([W_ie_0, W_ie_1, W_ie_2, W_ie_3], axis = 2)

        det_W = tf.linalg.det(W_ie)
        min_det = tf.reduce_min(tf.abs(det_W))
        max_det = tf.reduce_max(tf.abs(det_W))

        if min_det > self._lb and max_det < self._ub:
          iter += 1

          W_inv_ie = tf.linalg.inv(W_ie)
          c_ie = tf.matmul(W_inv_ie, o_bar2)

          min_c = tf.cast(tf.reduce_min(c_ie, axis = [-2, -1]) >= 0, tf.float32)
          unused_var, max_W_idx = tf.nn.top_k(min_c, k = 1)
          max_W_idx = tf.cast(tf.tile(max_W_idx, [1, 1, 4]), tf.float32)
          max_W_idx = tf.expand_dims(tf.expand_dims(tf.cast(tf.abs(max_W_idx - W_filter) < 0.1, tf.float32), axis = -1), axis = -1)
          max_W_inv = W_inv_ie * max_W_idx
          max_W_inv = tf.reduce_sum(max_W_inv, axis = 2)
          max_W = W_ie * max_W_idx
          max_W = tf.reduce_sum(max_W, axis = 2)

          u_ie = tf.matmul(tf.transpose(max_W_inv, [0, 1, 3, 2]), u0)
          u_norm = self._safe_norm(u_ie, axis = [-2, -1], keepdims = True)
          u_ie = tf.transpose(u_ie / u_norm, [0, 1, 3, 2])

          s1 = self._compute_spt(u_ie, i_v, i_p, i_t, get_h=False)
          s2 = self._compute_spt(-u_ie, j_v, j_p, j_t, get_h=False)
          s_bar = s2 - s1

          V_ie = tf.concat([max_W, tf.transpose(o_bar - s_bar, [0, 1, 3, 2])], axis = -1)
          
        else:
          continue

      loop_iter = iter

      s1 = self._compute_spt(u_ie, i_v, i_p, i_t, get_h=False)
      s2 = self._compute_spt(-u_ie, j_v, j_p, j_t, get_h=False)
      s_bar = s2 - s1

      "Growth Model"
      sig = tf.clip_by_value(self._safe_norm(o_bar, axis = -1) / self._safe_norm(o_bar - s_bar, axis = -1), clip_value_min = self._lb, clip_value_max = self._ub)
      var = tf.concat([tf.squeeze(u_ie, axis = 2), sig], axis = -1)

      tr_radius = 10.0 * tf.ones((tf.shape(vertices)[0], n_jc, 4, 1))
      g_iter = 0

      # noninvertible = False
      for j in range(10):
        f, jac = self._residual(o_bar, i_v, i_p, i_t, j_v, j_p, j_t, var)
        res = self._safe_norm(f, axis = -1)

        # jac_nan = tf.cast(tf.math.is_nan(tf.linalg.inv(jac)), tf.float32)
        # jac_nan = tf.reduce_max(jac_nan)
        # if jac_nan == 1.0 or noninvertible:
        #   noninvertible = True
        #   continue
        # else:
        #   noninvertible = False

        # try:
        #   jac_inv = tf.linalg.inv(jac)
        #   noninvertible = False
        # except:
        #   noninvertible = True
        #   continue

        if tf.reduce_max(res) > 1e-6:
          # # Debug 0
          # undef_j = tf.reduce_max(tf.cast(tf.is_nan(res), tf.float32))

          # cond = tf.cast(self._tf_cond(jac, 1e-10), tf.float32)
          # cond_min = tf.reduce_max(cond)
          # if cond_min < -0.1:
          #   undef = tf.tile(tf.expand_dims(tf.expand_dims(cond, axis = -1), axis = -1), [1, 1, 1, 4])
          #   undef = tf.concat([undef, jac], axis = 2)
          #   undef = tf.concat([undef, tf.tile(tf.expand_dims(tf.expand_dims(tf.linalg.det(jac), axis = -1), axis = -1), [1, 1, 1, 4])], axis = 2)
          #   undef = tf.concat([undef, tf.transpose(f, [0, 1, 3, 2])], axis = 2)
          #   undef = tf.concat([undef, tf.expand_dims(var, axis = 2)], axis = 2)
          #   undef = tf.concat([undef, tf.zeros_like(tf.expand_dims(var, axis = 2))], axis = 2)
          #   continue
          # else:
            # undef = tf.tile(tf.expand_dims(tf.expand_dims(cond, axis = -1), axis = -1), [1, 1, 1, 4])
          undef = jac
          undef = tf.concat([undef, tf.tile(tf.expand_dims(tf.expand_dims(tf.linalg.det(jac), axis = -1), axis = -1), [1, 1, 1, 4])], axis = 2)
          undef = tf.concat([undef, tf.transpose(f, [0, 1, 3, 2])], axis = 2)
          undef = tf.concat([undef, tf.expand_dims(var, axis = 2)], axis = 2)

          # jac_inv = tf.linalg.pinv(jac)
          # pseudo_id = tf.matmul(jac_inv, jac)
          # pseudo_diff = tf.reshape(tf.eye(4), [1, 1, 4, 4]) - pseudo_id
          # pseudo_diff = tf.reduce_max(tf.reduce_sum(tf.abs(pseudo_diff), axis = [-2, -1]))

          # undef = tf.concat([undef, tf.matmul(jac_inv, jac)], axis = 2)
          
          # if pseudo_diff < 0.2:

          jac_det = tf.linalg.det(jac)
          jac_det_min = tf.reduce_min(tf.abs(jac_det))
          jac_det_max = tf.reduce_max(tf.abs(jac_det))

          if jac_det_min > self._lb and jac_det_max < self._ub:
            undef = tf.concat([undef, tf.zeros_like(tf.expand_dims(var, axis = 2))], axis = 2)

            g_iter += 1
          
            jac_inv = tf.linalg.inv(jac)
            
            # try:
            #   jac_inv = tf.linalg.inv(jac)
            # except Exception:
            #   undef = tf.concat([undef, tf.ones_like(tf.expand_dims(var, axis = 2))], axis = 2)
            #   continue
            # undef = tf.concat([undef, tf.zeros_like(tf.expand_dims(var, axis = 2))], axis = 2)

            # jac_sgn = tf.sign(jac_inv)
            # jac_sgn = jac_sgn + tf.cast(jac_sgn == 0.0, tf.float32)
            # jac_inv = jac_sgn * tf.clip_by_value(tf.abs(jac_inv), clip_value_min = self._lb, clip_value_max = self._ub)

            # # Debug 1
            # undef_j = tf.concat([[undef_j], [tf.reduce_max(tf.cast(tf.is_nan(jac_inv), tf.float32))]], axis = -1)

            dN = -tf.matmul(jac_inv, f)
            grad = tf.matmul(tf.transpose(jac, [0, 1, 3, 2]), f)
            dC = - tf.clip_by_value((tf.reduce_sum(grad*grad, axis = [-2, -1], keepdims = True)) / (tf.reduce_sum(tf.matmul(jac, grad)*tf.matmul(jac, grad), axis = [-2, -1], keepdims = True) + self._lb), clip_value_min = self._lb, clip_value_max = self._ub) * grad
            dD = self._dogleg(dN, dC, tr_radius)

            # # Debug 2
            # undef_j = tf.concat([undef_j, [tf.reduce_max(tf.cast(tf.is_nan(tr_radius), tf.float32))]], axis = -1)
            # # Debug 3
            # undef_j = tf.concat([undef_j, [tf.reduce_max(tf.cast(tf.is_nan(tau), tf.float32))]], axis = -1)
            # # Debug 4
            # undef_j = tf.concat([undef_j, [tf.reduce_max(tf.cast(tf.is_nan(dD), tf.float32))]], axis = -1)

            f_next = self._residual(o_bar, i_v, i_p, i_t, j_v, j_p, j_t, var + tf.squeeze(dD, axis = -1), get_jac = False)

            # # Debug 5
            # undef_j = tf.concat([undef_j, [tf.reduce_max(tf.cast(tf.is_nan(f_next), tf.float32))]], axis = -1)

            actual_red = tf.expand_dims((tf.reduce_sum(f*f, axis = [-2, -1]) - tf.reduce_sum(f_next*f_next, axis = [-2, -1])), axis = -1) / 2.0
            predict_red = -tf.matmul(tf.transpose(grad, [0, 1, 3, 2]), dD) - tf.expand_dims(tf.expand_dims(tf.reduce_sum(tf.matmul(jac, dD)*tf.matmul(jac, dD), axis = [-2, -1]), axis = -1), axis = -1) / 2.0
            predict_red = tf.squeeze(predict_red, axis = -1)

            # # Debug 6
            # undef_j = tf.concat([undef_j, [tf.reduce_max(tf.cast(tf.is_nan(actual_red), tf.float32))]], axis = -1)

            # # Debug 7
            # undef_j = tf.concat([undef_j, [tf.reduce_max(tf.cast(tf.is_nan(predict_red), tf.float32))]], axis = -1)

            red_filter = predict_red == 0.0
            rho = tf.clip_by_value(tf.abs(actual_red) / tf.where(tf.abs(predict_red) < 1e-10, 1e-10*tf.ones(tf.shape(predict_red)), tf.abs(predict_red)), clip_value_min = self._lb, clip_value_max = self._ub)

            # # Debug 8
            # undef_j = tf.concat([undef_j, [tf.reduce_max(tf.cast(tf.is_nan(rho), tf.float32))]], axis = -1)

            large_rho = 1e+10 * tf.ones(tf.shape(rho))
            rho = tf.tile(tf.expand_dims(tf.where(red_filter, large_rho, rho), axis = -1), [1, 1, 4, 1])

            tr_filter1 = rho < 0.05
            tr_filter2 = rho > 0.9
            dD_norm = tf.tile(self._safe_norm(dD, axis = [-2, -1], keepdims = True), [1, 1, 4, 1])

            tr_radius = tf.where(tr_filter1, 0.25*dD_norm, tr_radius)
            tr_radius = tf.where(tr_filter2, tf.reduce_max(tf.concat([tr_radius, dD_norm], axis = -1), axis = -1, keepdims = True), tr_radius)

            var_filter = tf.squeeze(rho, axis = -1) > 0.05
            var = tf.where(var_filter, var + tf.squeeze(dD, axis = -1), var)

            # # Debug 9
            # undef_j = tf.concat([undef_j, [tf.reduce_max(tf.cast(tf.is_nan(var), tf.float32))]], axis = -1)

          else:
            undef = tf.concat([undef, tf.ones_like(tf.expand_dims(var, axis = 2))], axis = 2)
            # if iter == 10 and g_iter <= 1:
            #   u_ie = o_bar / self._safe_norm(o_bar, axis = -1, keepdims = True)
            #   s1 = self._compute_spt(u_ie, i_v, i_p, i_t, get_h=False)
            #   s2 = self._compute_spt(-u_ie, j_v, j_p, j_t, get_h=False)
            #   s_bar = s2 - s1
            #   sig = tf.clip_by_value(self._safe_norm(o_bar, axis = -1) / self._safe_norm(o_bar - s_bar, axis = -1), clip_value_min = self._lb, clip_value_max = self._ub)
            #   var = tf.concat([tf.squeeze(u_ie, axis = 2), sig], axis = -1)
            continue

        else:
          # undef_j = tf.constant([0.0])
          pass

        # undef_list.append(tf.reshape(undef_j, [1, -1]))


      "Record Growth Ratio"
      f = self._residual(o_bar, i_v, i_p, i_t, j_v, j_p, j_t, var, get_jac = False)

      u_ie = tf.expand_dims(var[:, :, :3], axis = 2)
      u_ie = u_ie / self._safe_norm(u_ie, axis = [-2, -1], keepdims = True)
      u_ie = tf.stop_gradient(u_ie)
      s1 = self._compute_spt(u_ie, i_v, i_p, i_t, get_h=False)
      s2 = self._compute_spt(-u_ie, j_v, j_p, j_t, get_h=False)
      s_bar = s2 - s1
      sigma = tf.clip_by_value(self._safe_norm(o_bar, axis = -1) / self._safe_norm(o_bar - s_bar, axis = -1), clip_value_min = 0.0, clip_value_max = self._ub_exponent)
      sigma = tf.reduce_min(sigma, axis = 1, keepdims = True)
      # sigma = tf.reduce_min(var[:, :, 3:4], axis = 1, keepdims = True)

      # if noninvertible:
      #   sigma = tf.ones(tf.shape(sigma)) * 0.9

      overlap_list.append(sigma)

      loop_iter = tf.concat([[loop_iter], [g_iter]], axis = -1)
      # loop_iter = g_iter


    "End of Loop"
    overlap = tf.concat(overlap_list, axis = 1)
    # undef = tf.concat(undef_list, axis = 0)
    # undef = tf.constant([0.0])
    return overlap, loop_iter, undef

  def _residual(self, o_bar, i_v, i_p, i_t, vertices, smoothness, translations, var, get_jac = True):
    x = tf.expand_dims(var[:, :, :3], axis = 2)

    s1, dsdx1 = self._compute_spt(x, i_v, i_p, i_t, get_h=False, get_dsdx=True)
    s2, dsdx2 = self._compute_spt(-x, vertices, smoothness, translations, get_h=False, get_dsdx=True)

    f = tf.concat([(var[:, :, 3:4]*tf.squeeze(s1 - s2, axis = 2) + (1 - var[:, :, 3:4])*tf.squeeze(-o_bar, axis = 2)), (tf.expand_dims(tf.reduce_sum(x*x, axis = [2, 3]) - 1.0, axis = -1))], axis = -1)
    f = tf.expand_dims(f, axis = -1)

    if get_jac:
      jac = tf.expand_dims(var[:, :, 3:4], axis = -1) * tf.squeeze(dsdx1 + dsdx2, axis = 2)
      jac = tf.concat([jac, tf.transpose(s1 - s2 + o_bar, [0, 1, 3, 2])], axis = -1)
      jac = tf.concat([jac, tf.concat([2.0*x, tf.zeros((tf.shape(vertices)[0], tf.shape(vertices)[1], 1, 1))], axis = -1)], axis = -2)
      jac = tf.clip_by_value(jac, clip_value_min = -self._ub, clip_value_max = self._ub)
      return f, jac
    else:
      return f

  def _dogleg(self, dN, dC, tr_radius):
    dN_norm = self._safe_norm(dN, axis = [-2, -1], keepdims = True)
    dC_norm = self._safe_norm(dC, axis = [-2, -1], keepdims = True)
    dN_filter = tf.cast(dN_norm < tr_radius, tf.float32)
    dC_filter = tf.cast(dC_norm > tr_radius, tf.float32)

    a = self._safe_norm(dN - dC, axis = [-2, -1], keepdims = True)
    a = tf.clip_by_value(a*a, clip_value_min = self._lb, clip_value_max = self._ub)
    b = tf.matmul(tf.transpose(dC, [0, 1, 3, 2]), dN - dC)
    c = b*b - a*(tf.reduce_sum(dC*dC, axis = [-2, -1], keepdims = True) - tr_radius*tr_radius)
    tau = tf.clip_by_value((-b + tf.sqrt(tf.abs(c) + self._lb)) / a, clip_value_min = -self._ub, clip_value_max = self._ub)

    dD = dN_filter * dN

    # dC_normalized = dC / self._safe_norm(dC, axis = [-2, -1], keepdims = True)
    # dC_sgn = tf.sign(dC_normalized)
    # dC_sgn = dC_sgn + tf.cast(dC_sgn == 0.0, tf.float32)

    dD += tf.cast((1.0 - dN_filter) == dC_filter, tf.float32) * dC / self._safe_norm(dC, axis = [-2, -1], keepdims = True)
    dD += tf.cast((1.0 - dN_filter) == (1.0 - dC_filter), tf.float32) * (dC + tau*(dN - dC))

    return tf.clip_by_value(dD, clip_value_min = -self._ub, clip_value_max = self._ub)

  def _safe_norm(self, x, axis, keepdims = False):
    norm = tf.sqrt(tf.reduce_sum(x*x, axis = axis, keepdims = keepdims) + self._lb)
    return norm
  
  # def _tf_cond(self, x, eps = 1e-10):
  #   s = tf.linalg.svd(x, compute_uv = False)
  #   # r = tf.clip_by_value(s, clip_value_min = 0.0, clip_value_max = self._ub) < eps

  #   r = s[..., 0] / s[..., -1]
  #   x_nan = tf.reduce_any(tf.is_nan(x), axis = [-2, -1])
  #   r_nan = tf.is_nan(r)
  #   r_inf = tf.fill(tf.shape(r), tf.constant(np.inf, r.dtype))
  #   tf.where(x_nan, r, tf.where(r_nan, r_inf, r))

  #   eps_inv = tf.cast(1 / eps, x.dtype)
  #   # return s, r
  #   return tf.is_finite(r) and (r < eps_inv)

  # def _compute_spt(self, x, mesh_idx, vertices, smoothness, translations):
  #   """Compute support function and surface point samples.

  #   Args:
  #     x: Tensor, [n_direction, dims], direction samples.
  #     vertices: Tensor, [batch_size, n_parts, n_vertices, dims], convex local vertices.
  #     smoothness: Tensor, [batch_size, n_parts], convex smoothness.
  #     translations: Tensor, [batch_size, n_parts, 1, dims], convex centers.

  #   Returns:
  #     points: Tensor, [batch_size, n_surface_points, dims], output surface point samples.
  #   """
  #   x = tf.transpose(x, [1, 0])
  #   x = tf.expand_dims(tf.expand_dims(x, axis = 0), axis = 0)
  #   x = tf.tile(x, [tf.shape(vertices)[0], tf.shape(vertices)[1], 1, 1])

  #   "z: Tensor, [batch_size, n_parts, n_vertices, n_out_points], dot product of v and x"
  #   z = tf.matmul(vertices, x)
  #   zm = tf.cast(z > 0, tf.float32) * z

  #   n_v = tf.shape(zm)[2]
  #   n_d = tf.shape(zm)[3]

  #   p = tf.expand_dims(tf.expand_dims(smoothness, axis = -1), axis = -1)
  #   p1 = tf.tile(p, [1, 1, n_v, n_d])
  #   p2 = tf.tile(p, [1, 1, 1, n_d])

  #   zm_log = tf.log(zm) / tf.log(10.0) # usually, =< 0
  #   zm_log = tf.reduce_max(zm_log, axis = 2, keepdims = True)
  #   exponent = zm_log * p2
  #   # uk = tf.cast(exponent > self._ub_exponent, tf.float32) * ((self._ub_exponent - exponent) / p2)
  #   lk = tf.cast(exponent < self._lb_exponent, tf.float32) * ((self._lb_exponent - exponent) / p2)
  #   k = tf.clip_by_value(tf.ceil(lk), clip_value_min = 0, clip_value_max = self._ub)
  #   base = tf.ones(tf.shape(k)) * 10.0
  #   k = tf.pow(base, k)
  #   k = tf.tile(k, [1, 1, n_v, 1])

  #   zm = zm * k

  #   zm_p = tf.clip_by_value(tf.pow(zm, p1), clip_value_min = self._lb, clip_value_max = self._ub) - tf.cast(zm == 0, tf.float32) * self._lb
  #   sum_zm_p = tf.reduce_sum(zm_p, axis = 2, keepdims = True)
  #   # sum_zm = tf.clip_by_value(tf.pow(sum_zm_p, 1 / p2 - 1), clip_value_min = 0, clip_value_max = self._ub)
  #   # sum_zm = tf.pow(sum_zm_p, 1 / p2 - 1)
  #   # sum_zm = tf.tile(sum_zm, [1, 1, n_v, 1])
  #   sum_zm = tf.clip_by_value(tf.pow(sum_zm_p, 1 / p2), clip_value_min = self._lb, clip_value_max = self._ub)
  #   sum_zm = tf.tile(sum_zm, [1, 1, n_v, 1])

  #   # dhdz = sum_zm * zm_p_1
  #   dhdz = tf.clip_by_value(tf.pow((zm / sum_zm), (p1 - 1)), clip_value_min = self._lb, clip_value_max = self._ub)
  #   dhdz_t = tf.transpose(dhdz, [0, 1, 3, 2])
  #   dhdx = tf.matmul(dhdz_t, vertices)

  #   points = dhdx + translations

  #   if self.ntop > 0:
  #     points = self._mesh_interpolation(points, mesh_idx, self._n_mesh_inter)
  #   points = self._remove_overlap(points)

  #   return points, dhdz, zm

  # def _mesh_interpolation(self, x, idx, n_mesh_inter):
  #   ntop = self.ntop
  #   nbottom = self.nbottom

  #   transform = tf.random.shuffle(tf.eye(self._n_mesh))

  #   mesh_vertices = tf.gather(x, idx, axis = 2)
  #   v1 = mesh_vertices[:, :, :, 0, :] - mesh_vertices[:, :, :, 1, :]
  #   v2 = mesh_vertices[:, :, :, 0, :] - mesh_vertices[:, :, :, 2, :]
  #   mesh_cross = tf.linalg.cross(v1, v2)
  #   mesh_area = tf.norm(mesh_cross, axis = -1) / 2
  #   mesh_area = tf.matmul(mesh_area, transform)

  #   mesh_area_top = tf.cast(mesh_area > tf.reduce_mean(mesh_area, axis = -1, keepdims = True), tf.float32)
  #   mesh_top_num = tf.cast(tf.reduce_min(tf.reduce_sum(mesh_area_top, axis = -1)), tf.int32)
  #   if ntop > mesh_top_num:
  #     ntop = mesh_top_num
  #   unused_var, mesh_topk = tf.nn.top_k(mesh_area_top, k = ntop)

  #   if nbottom > 0:
  #     mesh_area_bottom = tf.cast(mesh_area < tf.reduce_mean(mesh_area, axis = -1, keepdims = True), tf.float32)
  #     mesh_bottom_num = tf.cast(tf.reduce_min(tf.reduce_sum(mesh_area_bottom, axis = -1)), tf.int32)
  #     if nbottom > mesh_bottom_num:
  #       nbottom = mesh_bottom_num
  #     unused_var, mesh_bottomk = tf.nn.top_k(mesh_area_bottom, k = nbottom)

  #   # tf.math.reduce_std(mesh_area, axis = -1, keepdims = True)/3)

  #   mesh_filter = tf.tile(self.mesh_filter, [1, 1, 1, ntop + nbottom])
  #   mesh_filter = tf.cast(mesh_filter, tf.float32)

  #   if nbottom > 0:
  #     mesh_k = tf.concat([mesh_topk, mesh_bottomk], axis = -1)
  #   else:
  #     mesh_k = mesh_topk
  #   mesh_k = tf.expand_dims(mesh_k, axis = 2)
  #   mesh_k = tf.cast(tf.tile(mesh_k, [1, 1, self._n_mesh, 1]), tf.float32)
  #   mesh_idx = tf.cast(tf.abs(mesh_k - mesh_filter) < 0.5, tf.float32)
  #   mesh_idx = tf.matmul(transform, mesh_idx)

  #   mesh_idx = tf.expand_dims(tf.expand_dims(mesh_idx, axis = -1), axis = -1)

  #   mesh_vertices = tf.expand_dims(mesh_vertices, axis = 3)
  #   mesh_max = mesh_idx * mesh_vertices
  #   mesh_max = tf.reduce_sum(mesh_max, axis = 2)

  #   v1 = mesh_max[:, :, :, 0, :]
  #   v2 = mesh_max[:, :, :, 1, :]
  #   v3 = mesh_max[:, :, :, 2, :]

  #   mesh_points = []
  #   for i in range(n_mesh_inter):
  #     for j in range(n_mesh_inter):
  #       tmp1 = v1*(i+1)/n_mesh_inter + v2*(1 - (i+1)/n_mesh_inter)
  #       tmp2 = tmp1*(j+1)/n_mesh_inter + v3*(1 - (j+1)/n_mesh_inter)
  #       # point = tf.expand_dims(tmp2, axis = 2)
  #       mesh_points.append(tmp2)

  #   mesh_points = tf.concat(mesh_points, axis = 2)
  #   # mesh_points = tf.reduce_mean(mesh_vertices, axis = 3)

  #   points = tf.concat([x, mesh_points], axis = 2)
  #   return points

  # def _remove_overlap(self, x):
  #   points = tf.reshape(x, [tf.shape(x)[0], -1, tf.shape(x)[-1]])
  #   return points


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
