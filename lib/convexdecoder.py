from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

keras = tf.keras



class ConvexDecoder(keras.layers.Layer):
  """Differentiable shape rendering layer using multiple convex polytopes."""

  def __init__(self, n_parts, n_th1):
    super(ConvexDecoder, self).__init__()

    self._dims = 3
    self._n_parts = n_parts
    self._th1_range = np.pi / 2
    self._th2_range = np.pi

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


  def call(self, vertices, smoothness, pointcloud, nearsurf, out):
    """Decode the support function parameters into surface point samples.

    Args:
      vertices: Tensor, [batch_size, n_parts, n_vertices, 3], convex vertices.
      smoothness: Tensor, [batch_size, n_parts], convex smoothness.
      pointcloud: Tensor, [batch_size, n_points, 3], point cloud.

    Returns:
      points: Tensor, [batch_size, n_surface_points, 3], output surface point samples.
    """

    """
    mean_vertices: Tensor, [batch, n_parts, 1, 3], centor of geometry for each convex
    local_vertices: Tensor, [batch, n_parts, n_vertices, 3], vertices on convex local coordinate
    """
    mean_vertices = tf.reduce_mean(vertices, axis = 2, keepdims = True)
    local_vertices = vertices - mean_vertices

    # direction_h, points, overlap, distance, surf_distance, directions, iter, iter_dist, undef, undef_dist = self._compute_output(self.directions, local_vertices, smoothness, mean_vertices, pointcloud)
    overlap, distance, directions, surf_distance, surf_points, nearsurf_dist, out_dist = self._compute_output(self.directions, local_vertices, smoothness, mean_vertices, pointcloud, nearsurf, out)
    # mean_vertices = tf.reshape(mean_vertices, [tf.shape(mean_vertices)[0], tf.shape(mean_vertices)[1], tf.shape(mean_vertices)[-1]])

    # return points, direction_h, overlap, distance, surf_distance, directions, mean_vertices, iter, iter_dist, undef, undef_dist
    return overlap, distance, directions, surf_distance, surf_points, nearsurf_dist, out_dist

  def _compute_output(self, x, vertices, smoothness, translations, pointcloud, nearsurf, out):
    """Compute output.

    Args:
      x: Tensor, [n_direction, 3], direction samples.
      vertices: Tensor, [batch_size, n_parts, n_vertices, 3], convex local vertices.
      smoothness: Tensor, [batch_size, n_parts], convex smoothness.
      translations: Tensor, [batch_size, n_parts, 1, 3], convex centers.
      pointcloud: Tensor, [batch_size, n_points, 3] point cloud.

    Returns:
      direction_h: Tensor, [batch_size, n_parts, n_directions, 3 + 1], global directions and h(x).
      surf_points: Tensor, [batch_size, CD, 3], output surface point samples.
    """
    # local_directions = tf.expand_dims(tf.expand_dims(x, axis = 0), axis = 0)
    # local_directions = tf.tile(local_directions, [tf.shape(vertices)[0], tf.shape(vertices)[1], 1, 1])

    # x = tf.transpose(x, [0, 1])
    x = tf.expand_dims(tf.expand_dims(x, axis = 0), axis = 0)
    x = tf.tile(x, [tf.shape(vertices)[0], tf.shape(vertices)[1], 1, 1]) # (B,C,D,3)

    h = self._compute_spt(x, vertices, smoothness, translations, get_surf = False) # (B,C,D,1)
    # direction_h = tf.concat([local_directions, h], axis = -1)

    # overlap, iter, undef = self._compute_overlap(vertices, smoothness, translations)
    overlap = self._compute_overlap(vertices, smoothness, translations)
    # retraction, iter_ret, undef_ret, undef_ret_2 = self._compute_retraction(vertices, smoothness, translations, pointcloud)
    # distance, iter_dist, undef_dist = self._compute_distance(vertices, smoothness, translations, tf.tile(tf.expand_dims(pointcloud, axis = 1), [1, self._n_parts, 1, 1]))
    distance = self._compute_distance(vertices, smoothness, translations, tf.tile(tf.expand_dims(pointcloud, axis = 1), [1, self._n_parts, 1, 1]))

    surf_points, normals = self._compute_distance(vertices, smoothness, translations, (x*tf.clip_by_value(tf.reduce_max(h, axis = 2, keepdims = True), clip_value_min = 1e-10, clip_value_max = 10) + translations), get_points = True, get_normal = True) # (B,C,D,3)
    surf_points = tf.reshape(surf_points, [tf.shape(surf_points)[0], -1, 3]) # (B,CD,3)
    surf_points_2 = tf.tile(tf.expand_dims(surf_points, axis = 1), [1, self._n_parts, 1, 1]) # (B,C,CD,3)

    local_trans = tf.transpose(translations, [0, 2, 1, 3]) - translations # (B,C,C,3)
    normal_filter = tf.matmul(local_trans, tf.transpose(normals, [0, 1, 3, 2])) # (B,C,C,D)
    normal_filter = tf.reshape(normal_filter, [tf.shape(normal_filter)[0], self._n_parts, -1]) # (B,C,CD)
    normal_filter = tf.expand_dims(normal_filter, axis = -1) # (B,C,CD,1)
    normal_filter = normal_filter < 0

    # surf_distance, unused_var, unused_var2 = self._compute_distance(vertices, smoothness, translations, surf_points_2) # (B,C,CD,1)
    surf_distance = self._compute_distance(vertices, smoothness, translations, surf_points_2, normal_filter = normal_filter) # (B,C,CD,1)
    surf_distance = tf.reshape(surf_distance, [tf.shape(surf_distance)[0], self._n_parts, self._n_parts, -1, 1]) # (B,C,C,D,1)

    dist_filter = 100*tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.eye(self._n_parts), axis = -1), axis = -1), axis = 0) # (1,C,C,1,1)
    dist_filter = tf.tile(dist_filter, [tf.shape(surf_distance)[0], 1, 1, tf.shape(surf_distance)[3], 1])
    surf_distance = tf.where(dist_filter > 0, dist_filter, surf_distance)
    surf_distance = tf.reshape(surf_distance, [tf.shape(surf_distance)[0], self._n_parts, -1, 1])



    nearsurf_dist = self._compute_distance(vertices, smoothness, translations, tf.tile(tf.expand_dims(nearsurf, axis = 1), [1, self._n_parts, 1, 1]))
    out_dist = self._compute_distance(vertices, smoothness, translations, tf.tile(tf.expand_dims(out, axis = 1), [1, self._n_parts, 1, 1]))

    # retraction = tf.zeros((1, 100, 1))
    # iter_ret = tf.constant([1, 1])
    # undef_ret = tf.constant([[1.0], [1.0]])

    # return direction_h, surf_points, overlap, distance, surf_distance, x, iter, iter_dist, undef, undef_dist
    return overlap, distance, x, surf_distance, surf_points, nearsurf_dist, out_dist

  def _compute_spt(self, x, vertices, smoothness, translations, get_h = True, get_dsdx = False, get_surf = True):
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

    # zm_log = tf.log(zm) / tf.log(10.0)
    # zm_log = tf.reduce_max(zm_log, axis = 2, keepdims = True)
    # exponent = zm_log * p2
    # lk = tf.cast(exponent < self._lb_exponent, tf.float32) * ((-15 - exponent) / p2)
    # k = tf.clip_by_value(tf.ceil(lk), clip_value_min = 0.0, clip_value_max = self._ub_exponent)
    # base = tf.ones(tf.shape(k)) * 10.0
    # k = tf.pow(base, k)
    # k2 = k
    # k = tf.tile(k, [1, 1, n_v, 1])

    maxz = tf.reduce_max(z, axis = 2, keepdims = True) # (B,C,1,D)
    maxz = tf.clip_by_value(maxz, clip_value_min = 1e-30, clip_value_max = 1e+30)
    k = 1.0 / maxz
    k2 = k
    k = tf.tile(k, [1, 1, n_v, 1])

    zm = zm * k

    # zm_p = tf.clip_by_value(tf.pow(zm, p1), clip_value_min = self._lb, clip_value_max = self._ub) - tf.cast(zm == 0, tf.float32) * self._lb
    # sum_zm_p = tf.reduce_sum(zm_p, axis = 2, keepdims = True)
    # h = tf.clip_by_value(tf.pow(sum_zm_p, 1 / p2), clip_value_min = self._lb, clip_value_max = self._ub)
    # sum_zm = tf.tile(h, [1, 1, n_v, 1])

    zm_p = tf.pow(zm, p1)
    sum_zm_p = tf.reduce_sum(zm_p, axis = 2, keepdims = True)
    h = tf.pow(sum_zm_p, 1 / p2)
    sum_zm = tf.clip_by_value(tf.tile(h, [1, 1, n_v, 1]), clip_value_min = 1e-30, clip_value_max = 1e+30)


    "Compute dh/dx=s"
    dhdz = tf.clip_by_value(tf.pow((zm / sum_zm), (p1 - 1)), clip_value_min = 1e-30, clip_value_max = 1e+30)
    dhdz_t = tf.transpose(dhdz, [0, 1, 3, 2])
    dhdx = tf.matmul(dhdz_t, vertices)

    surf_points = dhdx + translations


    if get_dsdx:
      "Compute ds/dx"
      # zm_p_2 = tf.clip_by_value(tf.pow(zm, p1 - 2), clip_value_min = self._lb, clip_value_max = self._ub)
      zm_p_2 = tf.pow(zm, p1 - 2)
      zm_p_2 = tf.transpose(zm_p_2, [0, 1, 3, 2])
      diag_zm_p_2 = tf.linalg.diag(zm_p_2)

      # zm_p_1 = tf.clip_by_value(tf.pow(zm, p1 - 1), clip_value_min = self._lb, clip_value_max = self._ub)
      zm_p_1 = tf.pow(zm, p1 - 1)
      zm_p_1 = tf.transpose(zm_p_1, [0, 1, 3, 2])
      mat_zm_p_1 = tf.matmul(tf.expand_dims(zm_p_1, axis = -1), tf.expand_dims(zm_p_1, axis = -2))

      # sum_zm_1 = tf.clip_by_value(tf.pow(sum_zm_p, 1 / p2 - 1), clip_value_min = self._lb, clip_value_max = self._ub)
      # sum_zm_1 = tf.clip_by_value(tf.pow(tf.clip_by_value(sum_zm_p, clip_value_min = self._lb, clip_value_max = self._ub), 1 / p2 - 1), clip_value_min = self._lb, clip_value_max = self._ub)
      sum_zm_1 = tf.pow(sum_zm_p, 1 / p2 - 1)
      sum_zm_1 = tf.expand_dims(tf.transpose(sum_zm_1, [0, 1, 3, 2]), axis = -1)
      # sum_zm_2 = tf.clip_by_value(tf.pow(tf.clip_by_value(sum_zm_p, clip_value_min = 1e-10, clip_value_max = 1e+10), 1 / p2 - 2), clip_value_min = self._lb, clip_value_max = self._ub)
      # sum_zm_2 = tf.pow(tf.clip_by_value(sum_zm_p, clip_value_min = 1e-10, clip_value_max = 1e+10), 1 / p2 - 2)
      sum_zm_2 = tf.pow(sum_zm_p, 1 / p2 - 2)
      sum_zm_2 = tf.expand_dims(tf.transpose(sum_zm_2, [0, 1, 3, 2]), axis = -1)

      v_left = tf.tile(tf.expand_dims(tf.transpose(vertices, [0, 1, 3, 2]), axis = 2), [1, 1, n_d, 1, 1])
      v_right = tf.tile(tf.expand_dims(vertices, axis = 2), [1, 1, n_d, 1, 1])

      dsdx = diag_zm_p_2*sum_zm_1 - mat_zm_p_1*sum_zm_2

      dsdx = tf.matmul(tf.matmul(v_left, dsdx), v_right)


      p3 = tf.expand_dims(tf.expand_dims(tf.expand_dims(smoothness, axis = -1), axis = -1), axis = -1)
      k3 = tf.expand_dims(tf.transpose(k2, [0, 1, 3, 2]), axis = -1)

      dsdx = (p3 - 1.0) * k3 * dsdx
      dsdx = tf.clip_by_value(dsdx, clip_value_min = -1e+30, clip_value_max = 1e+30)

      if get_h:
        h = tf.clip_by_value(h / k2, clip_value_min = -1e+30, clip_value_max = 1e+30)
        h = tf.transpose(h, [0, 1, 3, 2])
        if get_surf:
          return h, surf_points, dsdx
        else:
          return h, dsdx
      else:
        if get_surf:
          return surf_points, dsdx
        else:
          return dsdx

    else:
      if get_h:
        h = tf.clip_by_value(h / k2, clip_value_min = -1e+30, clip_value_max = 1e+30)
        h = tf.transpose(h, [0, 1, 3, 2])
        if get_surf:
          return h, surf_points
        else:
          return h
      else:
        return surf_points

  # def _compute_overlap(self, vertices, smoothness, translations):
  #   """Compute overlapping state.

  #   Args:
  #     vertices: Tensor, [batch_size, n_parts, n_vertices, 3], convex local vertices.
  #     smoothness: Tensor, [batch_size, n_parts], convex smoothness.
  #     translations: Tensor, [batch_size, n_parts, 1, 3], convex centers.

  #   Returns:
  #     overlap: Tensor, [batch_size, n_parts, 1], convex minimum growth rate.
  #   """
  #   n_c = self._n_parts

  #   overlap_list = []
  #   # loop_iter = tf.constant([0])
  #   # undef_list = []
  #   # undef_j = tf.constant([0.0])
  #   # undef = tf.constant([0.0])

  #   for i in range(n_c - 1):
  #     n_jc = n_c - 1 - i
  #     i_v = tf.tile(vertices[:, i:(i+1), :, :], [1, n_jc, 1, 1])
  #     i_p = tf.tile(smoothness[:, i:(i+1)], [1, n_jc])
  #     i_t = tf.tile(translations[:, i:(i+1), :, :], [1, n_jc, 1, 1])

  #     j_v = vertices[:, i+1:, :, :]
  #     j_p = smoothness[:, i+1:]
  #     j_t = translations[:, i+1:, :, :]

  #     o_bar = j_t - translations[:, i:(i+1), :, :]
  #     o_bar2 = tf.tile(tf.expand_dims(tf.transpose(o_bar, [0, 1, 3, 2]), axis = 2), [1, 1, 4, 1, 1])

  #     u_ie = o_bar / self._safe_norm(o_bar, axis = -1, keepdims = True)
  #     u0 = tf.tile(tf.reshape(tf.constant([1.0, 1.0, 1.0]), [1, 1, 3, 1]), [tf.shape(vertices)[0], n_jc, 1, 1])


  #     "IE Initialization"
  #     V_ie = 1e-3 * tf.constant([[1.0, 0.0, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5], [0.0, 0.0, 1.0]])
  #     V_ie = tf.tile(tf.reshape(tf.transpose(V_ie, [1, 0]), [1, 1, 3, 4]), [tf.shape(vertices)[0], n_jc, 1, 1])
  #     W_ie = tf.tile(tf.zeros([1, 1, 4, 3, 3]), [tf.shape(vertices)[0], n_jc, 1, 1, 1])
  #     W_filter = tf.tile(tf.reshape(tf.constant([0.0, 1.0, 2.0, 3.0]), [1, 1, 4]), [tf.shape(vertices)[0], n_jc, 1])
  #     c_negative = tf.tile(tf.reshape(tf.constant([3.0, 3.0, 3.0, 3.0]), [1, 1, 4]), [tf.shape(vertices)[0], n_jc, 1])

  #     prev_W = W_ie

  #     # iter = 0
  #     for j in range(20):
  #       W_ie_0 = tf.expand_dims(tf.concat([V_ie[:, :, :, 1:2], V_ie[:, :, :, 2:3], V_ie[:, :, :, 3:4]], axis = -1), axis = 2)
  #       W_ie_1 = tf.expand_dims(tf.concat([V_ie[:, :, :, 0:1], V_ie[:, :, :, 2:3], V_ie[:, :, :, 3:4]], axis = -1), axis = 2)
  #       W_ie_2 = tf.expand_dims(tf.concat([V_ie[:, :, :, 0:1], V_ie[:, :, :, 1:2], V_ie[:, :, :, 3:4]], axis = -1), axis = 2)
  #       W_ie_3 = tf.expand_dims(tf.concat([V_ie[:, :, :, 0:1], V_ie[:, :, :, 1:2], V_ie[:, :, :, 2:3]], axis = -1), axis = 2)
  #       W_ie =  tf.concat([W_ie_0, W_ie_1, W_ie_2, W_ie_3], axis = 2) # (B,jC,4,3,3)

  #       det_W = tf.linalg.det(W_ie) # (B,jC,4)
  #       # min_det = tf.reduce_min(tf.abs(det_W))
  #       # max_det = tf.reduce_max(tf.abs(det_W))
  #       W_finite = tf.reduce_min(tf.cast(tf.is_finite(W_ie), tf.float32))

  #       det_filter = tf.tile(tf.expand_dims(tf.expand_dims(tf.abs(det_W), axis = -1), axis = -1), [1, 1, 1, 3, 3]) < 1e-30
  #       W_ie = tf.where(det_filter, prev_W, W_ie)
  #       prev_W = W_ie

  #       if W_finite > 0.5:
  #         # iter += 1

  #         # W_inv_ie = tf.linalg.inv(W_ie)
  #         W_inv_ie = self._cramer_inv(W_ie, 3)
  #         c_ie = tf.matmul(W_inv_ie, o_bar2)

  #         min_c = tf.cast(tf.reduce_min(c_ie, axis = [-2, -1]) >= 0, tf.float32)
  #         unused_var, max_W_idx = tf.nn.top_k(min_c, k = 1)
  #         max_W_idx = tf.cast(tf.tile(max_W_idx, [1, 1, 4]), tf.float32)

  #         c_filter = tf.tile(tf.reduce_max(min_c, axis = -1, keepdims = True), [1, 1, 4]) < 0.5
  #         max_W_idx = tf.where(c_filter, c_negative, max_W_idx)

  #         max_W_idx = tf.expand_dims(tf.expand_dims(tf.cast(tf.abs(max_W_idx - W_filter) < 0.1, tf.float32), axis = -1), axis = -1)
  #         max_W_inv = W_inv_ie * max_W_idx
  #         max_W_inv = tf.reduce_sum(max_W_inv, axis = 2)
  #         max_W = W_ie * max_W_idx
  #         max_W = tf.reduce_sum(max_W, axis = 2)

  #         u_ie = tf.matmul(tf.transpose(max_W_inv, [0, 1, 3, 2]), u0)
  #         u_norm = self._safe_norm(u_ie, axis = [-2, -1], keepdims = True)
  #         u_ie = tf.transpose(u_ie / u_norm, [0, 1, 3, 2])

  #         s1 = self._compute_spt(u_ie, i_v, i_p, i_t, get_h=False)
  #         s2 = self._compute_spt(-u_ie, j_v, j_p, j_t, get_h=False)
  #         s_bar = s2 - s1

  #         V_ie = tf.concat([max_W, tf.transpose(o_bar - s_bar, [0, 1, 3, 2])], axis = -1)
          
  #       else:
  #         continue

  #     # loop_iter = iter

  #     s1 = self._compute_spt(u_ie, i_v, i_p, i_t, get_h=False)
  #     s2 = self._compute_spt(-u_ie, j_v, j_p, j_t, get_h=False)
  #     s_bar = s2 - s1

  #     "Growth Model"
  #     sig = tf.clip_by_value(self._safe_norm(o_bar, axis = -1) / self._safe_norm(o_bar - s_bar, axis = -1), clip_value_min = self._lb, clip_value_max = self._ub)
  #     var = tf.concat([tf.squeeze(u_ie, axis = 2), sig], axis = -1)

  #     tr_radius = 10.0 * tf.ones((tf.shape(vertices)[0], n_jc, 4, 1))
  #     # g_iter = 0

  #     # noninvertible = False
  #     for j in range(10):
  #       f, jac = self._residual(o_bar, i_v, i_p, i_t, j_v, j_p, j_t, var)
  #       res = self._safe_norm(f, axis = [-2, -1])

  #       if tf.reduce_max(res) > 1e-6:
  #         # # Debug 0
  #         # undef_j = tf.reduce_max(tf.cast(tf.is_nan(res), tf.float32))

  #         # undef = jac
  #         # undef = tf.concat([undef, tf.tile(tf.expand_dims(tf.expand_dims(tf.linalg.det(jac), axis = -1), axis = -1), [1, 1, 1, 4])], axis = 2)
  #         # undef = tf.concat([undef, tf.transpose(f, [0, 1, 3, 2])], axis = 2)
  #         # undef = tf.concat([undef, tf.expand_dims(var, axis = 2)], axis = 2)

  #         # jac_det = tf.linalg.det(jac)
  #         # jac_det_min = tf.reduce_min(tf.abs(jac_det))
  #         # jac_det_max = tf.reduce_max(tf.abs(jac_det))

  #         # jac_finite = tf.reduce_min(tf.cast(tf.is_finite(jac), tf.float32))

  #         # undef = tf.concat([undef, tf.zeros_like(tf.expand_dims(var, axis = 2))], axis = 2)

  #         # g_iter += 1
        
  #         # jac_inv = tf.linalg.inv(jac)
  #         jac_inv = self._cramer_inv(jac, 4)

  #         # # Debug 1
  #         # undef_j = tf.concat([[undef_j], [tf.reduce_max(tf.cast(tf.is_nan(jac_inv), tf.float32))]], axis = -1)

  #         dN = -tf.matmul(jac_inv, f)
  #         grad = tf.matmul(tf.transpose(jac, [0, 1, 3, 2]), f)
  #         dC = - tf.clip_by_value((tf.reduce_sum(grad*grad, axis = [-2, -1], keepdims = True)) / (tf.reduce_sum(tf.matmul(jac, grad)*tf.matmul(jac, grad), axis = [-2, -1], keepdims = True) + self._lb), clip_value_min = self._lb, clip_value_max = self._ub) * grad
  #         dD = self._dogleg(dN, dC, tr_radius)

  #         # # Debug 2
  #         # undef_j = tf.concat([undef_j, [tf.reduce_max(tf.cast(tf.is_nan(tr_radius), tf.float32))]], axis = -1)
  #         # # Debug 3
  #         # undef_j = tf.concat([undef_j, [tf.reduce_max(tf.cast(tf.is_nan(tau), tf.float32))]], axis = -1)
  #         # # Debug 4
  #         # undef_j = tf.concat([undef_j, [tf.reduce_max(tf.cast(tf.is_nan(dD), tf.float32))]], axis = -1)

  #         f_next = self._residual(o_bar, i_v, i_p, i_t, j_v, j_p, j_t, var + tf.squeeze(dD, axis = -1), get_jac = False)

  #         # # Debug 5
  #         # undef_j = tf.concat([undef_j, [tf.reduce_max(tf.cast(tf.is_nan(f_next), tf.float32))]], axis = -1)

  #         actual_red = tf.expand_dims((tf.reduce_sum(f*f, axis = [-2, -1]) - tf.reduce_sum(f_next*f_next, axis = [-2, -1])), axis = -1) / 2.0
  #         predict_red = -tf.matmul(tf.transpose(grad, [0, 1, 3, 2]), dD) - tf.expand_dims(tf.expand_dims(tf.reduce_sum(tf.matmul(jac, dD)*tf.matmul(jac, dD), axis = [-2, -1]), axis = -1), axis = -1) / 2.0
  #         predict_red = tf.squeeze(predict_red, axis = -1)

  #         # # Debug 6
  #         # undef_j = tf.concat([undef_j, [tf.reduce_max(tf.cast(tf.is_nan(actual_red), tf.float32))]], axis = -1)

  #         # # Debug 7
  #         # undef_j = tf.concat([undef_j, [tf.reduce_max(tf.cast(tf.is_nan(predict_red), tf.float32))]], axis = -1)

  #         red_filter = predict_red == 0.0
  #         rho = actual_red / tf.abs(predict_red)
  #         # # Debug 8
  #         # undef_j = tf.concat([undef_j, [tf.reduce_max(tf.cast(tf.is_nan(rho), tf.float32))]], axis = -1)

  #         large_rho = 1e+10 * tf.ones(tf.shape(rho))
  #         rho = tf.tile(tf.expand_dims(tf.where(red_filter, large_rho, rho), axis = -1), [1, 1, 4, 1])

  #         tr_filter1 = rho < 0.05
  #         tr_filter2 = rho > 0.9
  #         dD_norm = tf.tile(self._safe_norm(dD, axis = [-2, -1], keepdims = True), [1, 1, 4, 1])

  #         tr_radius = tf.where(tr_filter1, 0.25*dD_norm, tr_radius)
  #         tr_radius = tf.where(tr_filter2, tf.reduce_max(tf.concat([tr_radius, dD_norm], axis = -1), axis = -1, keepdims = True), tr_radius)

  #         var_filter = tf.squeeze(rho, axis = -1) > 0.05
  #         var = tf.where(var_filter, var + tf.squeeze(dD, axis = -1), var)

  #         # # Debug 9
  #         # undef_j = tf.concat([undef_j, [tf.reduce_max(tf.cast(tf.is_nan(var), tf.float32))]], axis = -1)

  #       else:
  #         # undef = jac
  #         # undef = tf.concat([undef, tf.tile(tf.expand_dims(tf.expand_dims(tf.linalg.det(jac), axis = -1), axis = -1), [1, 1, 1, 4])], axis = 2)
  #         # undef = tf.concat([undef, tf.transpose(f, [0, 1, 3, 2])], axis = 2)
  #         # undef = tf.concat([undef, tf.expand_dims(var, axis = 2)], axis = 2)
  #         # undef = tf.concat([undef, tf.zeros_like(tf.expand_dims(var, axis = 2))], axis = 2)
  #         # undef_j = tf.constant([0.0])
  #         pass

  #       # undef_list.append(tf.reshape(undef_j, [1, -1]))


  #     "Record Growth Ratio"
  #     f = self._residual(o_bar, i_v, i_p, i_t, j_v, j_p, j_t, var, get_jac = False)

  #     u_ie = tf.expand_dims(var[:, :, :3], axis = 2)
  #     u_ie = u_ie / self._safe_norm(u_ie, axis = [-2, -1], keepdims = True)
  #     u_ie = tf.stop_gradient(u_ie)
  #     s1 = self._compute_spt(u_ie, i_v, i_p, i_t, get_h=False)
  #     s2 = self._compute_spt(-u_ie, j_v, j_p, j_t, get_h=False)
  #     s_bar = s2 - s1
  #     sigma = tf.clip_by_value(self._safe_norm(o_bar, axis = -1) / self._safe_norm(o_bar - s_bar, axis = -1), clip_value_min = 0.0, clip_value_max = self._ub_exponent)
  #     sigma = tf.reduce_min(sigma, axis = 1, keepdims = True)
  #     # sigma = tf.reduce_min(var[:, :, 3:4], axis = 1, keepdims = True)

  #     # if noninvertible:
  #     #   sigma = tf.ones(tf.shape(sigma)) * 0.9

  #     # gap_filter = tf.squeeze(tf.matmul(o_bar, tf.transpose(s_bar, [0, 1, 3, 2])), axis = -1) > 0 # (B,C,1)
  #     # gap = self._safe_norm(s_bar, axis = -1) # (B,C,1)
  #     # gap = tf.where(gap_filter, gap, -gap)

  #     overlap_list.append(sigma)

  #     # loop_iter = tf.concat([[loop_iter], [g_iter]], axis = -1)
  #     # loop_iter = g_iter


  #   "End of Loop"
  #   overlap = tf.concat(overlap_list, axis = 1)
  #   # undef = tf.concat(undef_list, axis = 0)
  #   # undef = tf.constant([0.0])
  #   # return overlap, loop_iter, undef
  #   return overlap

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

    eye_filter = tf.reshape(tf.eye(n_c), [1, n_c, n_c, 1, 1])
    eye_filter_2 = tf.tile(eye_filter, [tf.shape(vertices)[0], 1, 1, 1, 1]) # (B,C,C,1,1)
    eye_filter = tf.tile(eye_filter, [tf.shape(vertices)[0], 1, 1, 3, 1]) # (B,C,C,3,1)

    # i_v = tf.tile(vertices[:, i:(i+1), :, :], [1, n_c, 1, 1])
    # i_p = tf.tile(smoothness[:, i:(i+1)], [1, n_c])
    # i_t = tf.tile(translations[:, i:(i+1), :, :], [1, n_c, 1, 1])

    # j_v = vertices[:, i+1:, :, :]
    # j_p = smoothness[:, i+1:]
    # j_t = translations[:, i+1:, :, :]

    # o_bar = j_t - translations[:, i:(i+1), :, :]
    # o_bar2 = tf.tile(tf.expand_dims(tf.transpose(o_bar, [0, 1, 3, 2]), axis = 2), [1, 1, 4, 1, 1])
    o_bar = tf.expand_dims(tf.transpose(translations, [0, 2, 1, 3]) - translations, axis = -1) # (B,C,C,3,1)
    o_bar2 = tf.tile(tf.expand_dims(o_bar, axis = 3), [1, 1, 1, 4, 1, 1]) # (B,C,C,4,3,1)

    u_ie = tf.where(eye_filter > 0.5, tf.ones_like(o_bar), o_bar)
    u_ie = u_ie / self._safe_norm(u_ie, axis = -1, keepdims = True) # (B,C,C,3,1)
    u_ie = tf.squeeze(u_ie, axis = -1)
    u0 = tf.tile(tf.reshape(tf.constant([1.0, 1.0, 1.0]), [1, 1, 1, 3, 1]), [tf.shape(vertices)[0], n_c, n_c, 1, 1]) # (B,C,C,3,1)


    "IE Initialization"
    V_ie = 1e-3 * tf.constant([[1.0, 0.0, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5], [0.0, 0.0, 1.0]])
    V_ie = tf.tile(tf.reshape(tf.transpose(V_ie, [1, 0]), [1, 1, 1, 3, 4]), [tf.shape(vertices)[0], n_c, n_c, 1, 1]) # (B,C,C,3,4)
    W_ie = tf.tile(tf.zeros([1, 1, 1, 4, 3, 3]), [tf.shape(vertices)[0], n_c, n_c, 1, 1, 1]) # (B,C,C,4,3,3)
    W_filter = tf.tile(tf.reshape(tf.constant([0.0, 1.0, 2.0, 3.0]), [1, 1, 1, 4]), [tf.shape(vertices)[0], n_c, n_c, 1])
    c_negative = tf.tile(tf.reshape(tf.constant([3.0, 3.0, 3.0, 3.0]), [1, 1, 1, 4]), [tf.shape(vertices)[0], n_c, n_c, 1])

    prev_W = W_ie

    # iter = 0
    for j in range(20):
      W_ie_0 = tf.expand_dims(tf.concat([V_ie[:, :, :, :, 1:2], V_ie[:, :, :, :, 2:3], V_ie[:, :, :, :, 3:4]], axis = -1), axis = 3)
      W_ie_1 = tf.expand_dims(tf.concat([V_ie[:, :, :, :, 0:1], V_ie[:, :, :, :, 2:3], V_ie[:, :, :, :, 3:4]], axis = -1), axis = 3)
      W_ie_2 = tf.expand_dims(tf.concat([V_ie[:, :, :, :, 0:1], V_ie[:, :, :, :, 1:2], V_ie[:, :, :, :, 3:4]], axis = -1), axis = 3)
      W_ie_3 = tf.expand_dims(tf.concat([V_ie[:, :, :, :, 0:1], V_ie[:, :, :, :, 1:2], V_ie[:, :, :, :, 2:3]], axis = -1), axis = 3)
      W_ie =  tf.concat([W_ie_0, W_ie_1, W_ie_2, W_ie_3], axis = 3) # (B,C,C,4,3,3)

      det_W = tf.linalg.det(W_ie) # (B,C,C,4)
      W_finite = tf.reduce_min(tf.cast(tf.is_finite(W_ie), tf.float32))

      det_filter = tf.tile(tf.expand_dims(tf.expand_dims(tf.abs(det_W), axis = -1), axis = -1), [1, 1, 1, 1, 3, 3]) < 1e-30
      W_ie = tf.where(det_filter, prev_W, W_ie)
      prev_W = W_ie

      if W_finite > 0.5:
        W_inv_ie = self._cramer_inv(W_ie, 3)
        c_ie = tf.matmul(W_inv_ie, o_bar2) # (B,C,C,4,3,1)

        min_c = tf.cast(tf.reduce_min(c_ie, axis = [-2, -1]) >= 0, tf.float32)
        unused_var, max_W_idx = tf.nn.top_k(min_c, k = 1)
        max_W_idx = tf.cast(tf.tile(max_W_idx, [1, 1, 1, 4]), tf.float32) # (B,C,C,4)

        c_filter = tf.tile(tf.reduce_max(min_c, axis = -1, keepdims = True), [1, 1, 1, 4]) < 0.5
        max_W_idx = tf.where(c_filter, c_negative, max_W_idx)

        max_W_idx = tf.expand_dims(tf.expand_dims(tf.cast(tf.abs(max_W_idx - W_filter) < 0.1, tf.float32), axis = -1), axis = -1)
        max_W_inv = W_inv_ie * max_W_idx
        max_W_inv = tf.reduce_sum(max_W_inv, axis = 3) # (B,C,C,3,3)
        max_W = W_ie * max_W_idx
        max_W = tf.reduce_sum(max_W, axis = 3) # (B,C,C,3,3)

        u_ie = tf.matmul(tf.transpose(max_W_inv, [0, 1, 2, 4, 3]), u0) # (B,C,C,3,1)
        u_norm = self._safe_norm(u_ie, axis = [-2, -1], keepdims = True)
        u_ie = tf.squeeze(u_ie / u_norm, axis = -1) # (B,C,C,3)

        s1 = self._compute_spt(u_ie, vertices, smoothness, translations, get_h=False)
        s2 = self._compute_spt(tf.transpose(-u_ie, [0, 2, 1, 3]), vertices, smoothness, translations, get_h=False) # (B,C,C,3)
        s_bar = tf.expand_dims(tf.transpose(s2, [0, 2, 1, 3]) - s1, axis = -1) # (B,C,C,3,1)

        V_ie = tf.concat([max_W, o_bar - s_bar], axis = -1)
        
      else:
        continue

    # loop_iter = iter

    s1 = self._compute_spt(u_ie, translations, smoothness, translations, get_h=False)
    s2 = self._compute_spt(tf.transpose(-u_ie, [0, 2, 1, 3]), translations, smoothness, translations, get_h=False)
    s_bar = tf.expand_dims(tf.transpose(s2, [0, 2, 1, 3]) - s1, axis = -1) # (B,C,C,3,1)

    "Growth Model"
    sig = tf.clip_by_value(self._safe_norm(o_bar, axis = -2) / self._safe_norm(o_bar - s_bar, axis = -2), clip_value_min = self._lb, clip_value_max = self._ub)
    var = tf.concat([u_ie, sig], axis = -1) # (B,C,C,4)

    tr_radius = 10.0 * tf.ones((tf.shape(vertices)[0], n_c, n_c, 4, 1))

    for j in range(10):
      f, jac = self._residual(o_bar, vertices, smoothness, translations, var) # (B,C,C,4,1), (B,C,C,4,4)
      res = self._safe_norm(f, axis = [-2, -1])

      if tf.reduce_max(res) > 1e-6:

        jac_inv = self._cramer_inv(jac, 4)

        dN = -tf.matmul(jac_inv, f) # (B,C,C,4,1)
        grad = tf.matmul(tf.transpose(jac, [0, 1, 2, 4, 3]), f)
        dC = - tf.clip_by_value((tf.reduce_sum(grad*grad, axis = [-2, -1], keepdims = True)) / (tf.reduce_sum(tf.matmul(jac, grad)*tf.matmul(jac, grad), axis = [-2, -1], keepdims = True) + self._lb), clip_value_min = self._lb, clip_value_max = self._ub) * grad
        dD = self._dogleg_ret(dN, dC, tr_radius)

        f_next = self._residual(o_bar, vertices, smoothness, translations, var + tf.squeeze(dD, axis = -1), get_jac = False)

        actual_red = tf.expand_dims((tf.reduce_sum(f*f, axis = [-2, -1]) - tf.reduce_sum(f_next*f_next, axis = [-2, -1])), axis = -1) / 2.0
        predict_red = -tf.matmul(tf.transpose(grad, [0, 1, 2, 4, 3]), dD) - tf.reduce_sum(tf.matmul(jac, dD)*tf.matmul(jac, dD), axis = [-2, -1], keepdims = True) / 2.0
        predict_red = tf.squeeze(predict_red, axis = -1) # (B,C,C,1)

        red_filter = predict_red == 0.0
        rho = actual_red / tf.abs(predict_red)

        large_rho = 1e+10 * tf.ones(tf.shape(rho))
        rho = tf.tile(tf.expand_dims(tf.where(red_filter, large_rho, rho), axis = -1), [1, 1, 1, 4, 1])

        tr_filter1 = rho < 0.05
        tr_filter2 = rho > 0.9
        dD_norm = tf.tile(self._safe_norm(dD, axis = [-2, -1], keepdims = True), [1, 1, 1, 4, 1])

        tr_radius = tf.where(tr_filter1, 0.25*dD_norm, tr_radius)
        tr_radius = tf.where(tr_filter2, tf.reduce_max(tf.concat([tr_radius, 3*dD_norm], axis = -1), axis = -1, keepdims = True), tr_radius)

        var_filter = tf.squeeze(rho, axis = -1) > 0.05
        var = tf.where(var_filter, var + tf.squeeze(dD, axis = -1), var)


    "Record Growth Ratio"
    # f = self._residual(o_bar, vertices, smoothness, translations, var, get_jac = False)

    u_ie = var[:, :, :, :3]
    u_ie = tf.where(tf.squeeze(eye_filter, axis = -1) > 0.5, tf.ones_like(u_ie), u_ie)
    u_ie = u_ie / self._safe_norm(u_ie, axis = -1, keepdims = True)
    u_ie = tf.stop_gradient(u_ie)
    s1 = self._compute_spt(u_ie, vertices, smoothness, translations, get_h=False)
    s2 = self._compute_spt(tf.transpose(-u_ie, [0, 2, 1, 3]), vertices, smoothness, translations, get_h=False)
    s_bar = tf.expand_dims(tf.transpose(s2, [0, 2, 1, 3]) - s1, axis = -1) # (B,C,C,3,1)
    sigma = tf.clip_by_value(self._safe_norm(o_bar, axis = -2) / self._safe_norm(o_bar - s_bar, axis = -2), clip_value_min = 0.0, clip_value_max = self._ub_exponent)
    sigma = tf.where(tf.squeeze(eye_filter_2, axis = -1) > 0.5, 1e+10*tf.ones_like(sigma), sigma) # (B,C,C,1)
    sigma = tf.squeeze(sigma, axis = -1) # (B,C,C)
    # sigma = tf.linalg.band_part(sigma, 0, -1) + 1e+10*tf.linalg.band_part(tf.ones_like(sigma), -1, 0)
    # sigma = tf.reduce_min(sigma[:, :, :-1], axis = 2, keepdims = True) # (B,C,1)
    sigma = tf.reduce_min(sigma, axis = 2, keepdims = True) # (B,C,1)


    "End of Loop"
    overlap = sigma

    return overlap


  # def _compute_retraction(self, vertices, smoothness, translations, points):
  #   """Compute retarction state.

  #   Args:
  #     vertices: Tensor, [batch_size, n_parts, n_vertices, 3], convex local vertices.
  #     smoothness: Tensor, [batch_size, n_parts], convex smoothness.
  #     translations: Tensor, [batch_size, n_parts, 1, 3], convex centers.
  #     points: Tensor, [batch_size, n_points, 3], point clouds.

  #   Returns:
  #     retraction: Tensor, [batch_size, n_points, 1], minimum growth rate for each point.
  #   """
  #   n_c = self._n_parts

  #   undef_ret = tf.constant([[1.0], [1.0]])
  #   loop_iter = tf.constant([0])
  #   tr_list = []

  #   sp = tf.tile(tf.expand_dims(tf.expand_dims(points, axis = 1), axis = 3), [1, n_c, 1, 1, 1]) # (B,C,P,1,3)
  #   o = tf.tile(tf.expand_dims(translations, axis = 2), [1, 1, tf.shape(points)[1], 1, 1]) # (B,C,P,1,3)
  #   o_bar = sp - o
  #   o_bar2 = tf.tile(tf.expand_dims(tf.transpose(o_bar, [0, 1, 2, 4, 3]), axis = 3), [1, 1, 1, 4, 1, 1]) # (B,C,P,4,3,1)

  #   u_ie = o_bar / tf.linalg.norm(o_bar, axis = -1, keepdims = True)
  #   u_ie = tf.squeeze(u_ie, axis = 3) # (B,C,P,3)
  #   s = tf.zeros_like(u_ie)
  #   s = tf.expand_dims(s, axis = 3) # (B,C,P,1,3)
  #   u0 = tf.tile(tf.reshape(tf.constant([1.0, 1.0, 1.0]), [1, 1, 1, 3, 1]), [tf.shape(vertices)[0], n_c, tf.shape(points)[1], 1, 1]) # (B,C,P,3,1)


  #   "IE Initialization"
  #   V_ie = 1e-3 * tf.constant([[1.0, 0.0, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5], [0.0, 0.0, 1.0]])
  #   V_ie = tf.tile(tf.reshape(tf.transpose(V_ie, [1, 0]), [1, 1, 1, 3, 4]), [tf.shape(vertices)[0], n_c, tf.shape(points)[1], 1, 1]) # (B,C,P,3,4)
  #   W_ie = tf.tile(tf.zeros([1, 1, 1, 4, 3, 3]), [tf.shape(vertices)[0], n_c, tf.shape(points)[1], 1, 1, 1]) # (B,C,P,4,3,3)
  #   W_filter = tf.tile(tf.reshape(tf.constant([0.0, 1.0, 2.0, 3.0]), [1, 1, 1, 4]), [tf.shape(vertices)[0], n_c, tf.shape(points)[1], 1]) # (B,C,P,4)
  #   c_negative = tf.tile(tf.reshape(tf.constant([3.0, 3.0, 3.0, 3.0]), [1, 1, 1, 4]), [tf.shape(vertices)[0], n_c, tf.shape(points)[1], 1]) # (B,C,P,4)

  #   prev_W = W_ie
  #   max_W = tf.tile(tf.zeros_like(u0), [1, 1, 1, 1, 3])
  #   max_W_inv = tf.tile(tf.zeros_like(u0), [1, 1, 1, 1, 3])

  #   iter = 0
  #   for j in range(30):
  #     W_ie_0 = tf.expand_dims(tf.concat([V_ie[:, :, :, :, 1:2], V_ie[:, :, :, :, 2:3], V_ie[:, :, :, :, 3:4]], axis = -1), axis = 3) # (B,C,P,1,3,3)
  #     W_ie_1 = tf.expand_dims(tf.concat([V_ie[:, :, :, :, 0:1], V_ie[:, :, :, :, 2:3], V_ie[:, :, :, :, 3:4]], axis = -1), axis = 3)
  #     W_ie_2 = tf.expand_dims(tf.concat([V_ie[:, :, :, :, 0:1], V_ie[:, :, :, :, 1:2], V_ie[:, :, :, :, 3:4]], axis = -1), axis = 3)
  #     W_ie_3 = tf.expand_dims(tf.concat([V_ie[:, :, :, :, 0:1], V_ie[:, :, :, :, 1:2], V_ie[:, :, :, :, 2:3]], axis = -1), axis = 3)
  #     W_ie =  tf.concat([W_ie_0, W_ie_1, W_ie_2, W_ie_3], axis = 3) # (B,C,P,4,3,3)

  #     det_W = tf.linalg.det(W_ie) # (B,C,P,4)
  #     min_det = tf.reduce_min(tf.abs(det_W))
  #     max_det = tf.reduce_max(tf.abs(det_W))
  #     W_finite = tf.reduce_min(tf.cast(tf.is_finite(W_ie), tf.float32))

  #     det_filter = tf.tile(tf.expand_dims(tf.expand_dims(tf.abs(det_W), axis = -1), axis = -1), [1, 1, 1, 1, 3, 3]) < 1e-30
  #     W_ie = tf.where(det_filter, prev_W, W_ie)
  #     prev_W = W_ie

  #     if (W_finite > 0.5):
  #       iter += 1

  #       # W_inv_ie = tf.linalg.inv(W_ie)
  #       W_inv_ie = self._cramer_inv(W_ie, 3) # (B,C,P,4,3,3)
  #       c_ie = tf.matmul(W_inv_ie, o_bar2) # (B,C,P,4,3,1)

  #       min_c = tf.cast(tf.reduce_min(c_ie, axis = [-2, -1]) >= 0, tf.float32) # (B,C,P,4)
  #       unused_var, max_W_idx = tf.nn.top_k(min_c, k = 1)
  #       max_W_idx = tf.cast(tf.tile(max_W_idx, [1, 1, 1, 4]), tf.float32) # (B,C,P,4)

  #       c_filter = tf.tile(tf.reduce_max(min_c, axis = -1, keepdims = True), [1, 1, 1, 4]) < 0.5
  #       max_W_idx = tf.where(c_filter, c_negative, max_W_idx)

  #       max_W_idx = tf.expand_dims(tf.expand_dims(tf.cast(tf.abs(max_W_idx - W_filter) < 0.1, tf.float32), axis = -1), axis = -1) # (B,C,P,4,1,1)
  #       max_W_inv = W_inv_ie * max_W_idx # (B,C,P,4,3,3)
  #       max_W_inv = tf.reduce_sum(max_W_inv, axis = 3) # (B,C,P,3,3)
  #       max_W = W_ie * max_W_idx
  #       max_W = tf.reduce_sum(max_W, axis = 3) # (B,C,P,3,3)

  #       u_ie = tf.matmul(tf.transpose(max_W_inv, [0, 1, 2, 4, 3]), u0) # (B,C,P,3,1)
  #       u_norm = self._safe_norm(u_ie, axis = [-2, -1], keepdims = True) # (B,C,P,1,1)
  #       u_ie = tf.transpose(u_ie / u_norm, [0, 1, 2, 4, 3]) # (B,C,P,1,3)
  #       u_ie = tf.squeeze(u_ie, axis = 3) # (B,C,P,3)

  #       s = self._compute_spt(u_ie, vertices, smoothness, translations, get_h=False) # (B,C,P,3)
  #       s = tf.expand_dims(s, axis = 3) # (B,C,P,1,3)

  #       V_ie = tf.concat([max_W, tf.transpose(s - o, [0, 1, 2, 4, 3])], axis = -1) # (B,C,P,3,4)
        
  #     else:
  #       pass
      
  #     tr_list.append(tf.expand_dims(tf.concat([max_W, max_W_inv, tf.expand_dims(u_ie, axis = -1), tf.transpose(s, [0, 1, 2, 4, 3])], axis = -1), axis = -1)) # (B,C,P,3,3+3+1+1, 1)

  #   loop_iter = iter

  #   s = self._compute_spt(u_ie, vertices, smoothness, translations, get_h=False) # (B,C,P,3)
  #   s = tf.expand_dims(s, axis = 3) # (B,C,P,1,3)

  #   "Growth Model"
  #   sig = tf.clip_by_value(tf.linalg.norm(o_bar, axis = -1) / tf.linalg.norm(s - o, axis = -1), clip_value_min = self._lb, clip_value_max = self._ub) # (B,C,P,1)
  #   var = tf.concat([u_ie, sig], axis = -1) # (B,C,P,4)

  #   tr_radius = 10.0 * tf.ones((tf.shape(vertices)[0], n_c, tf.shape(points)[1], 4, 1)) # (B,C,P,4,1)
  #   tr_acc = tf.zeros_like(tr_radius)
  #   g_iter = 0
  #   g_res_iter = 0

  #   initial_u = tf.expand_dims(tf.concat([u_ie, tf.zeros_like(sig)], axis = -1), axis = -1)

  #   for j in range(20):
  #     f, jac = self._residual_ret(sp, o, vertices, smoothness, translations, var) # (B,C,P,4,1), (B,C,P,4,4)
  #     res = tf.linalg.norm(f, axis = [-2, -1]) # (B,C,P)

  #     jac_det = tf.linalg.det(jac) # (B,C,P)
  #     # jac_det_min = tf.reduce_min(tf.abs(jac_det))
  #     # jac_det_max = tf.reduce_max(tf.abs(jac_det))
  #     # jac_finite = tf.reduce_min(tf.cast(tf.is_finite(jac), tf.float32))

  #     jac_finite_filter = tf.is_finite(tf.tile(tf.expand_dims(jac_det, axis = -1), [1, 1, 1, 4]))

  #     dD = tf.ones_like(tr_radius)
  #     dD_norm = tf.ones_like(dD)
  #     dN = tf.ones_like(tr_radius)
  #     dC = tf.ones_like(tr_radius)
  #     jac_inv = tf.ones_like(jac)
  #     act_var = tf.ones_like(tr_radius)
  #     pred_var = tf.ones_like(tr_radius)
      
  #     if tf.reduce_max(res) > 1e-6:
  #       g_res_iter += 1

  #       g_iter += 1
      
  #       # jac_inv = tf.linalg.inv(jac)
  #       jac_inv = self._cramer_inv(jac, 4) # (B,C,P,4,4)

  #       dN = -tf.matmul(jac_inv, f) # (B,C,P,4,1)
  #       grad = tf.matmul(tf.transpose(jac, [0, 1, 2, 4, 3]), f) # (B,C,P,4,1)
  #       dC = - tf.clip_by_value((tf.reduce_sum(grad*grad, axis = [-2, -1], keepdims = True)) / (tf.reduce_sum(tf.matmul(jac, grad)*tf.matmul(jac, grad), axis = [-2, -1], keepdims = True) + self._lb), clip_value_min = self._lb, clip_value_max = self._ub) * grad
  #       dD = self._dogleg_ret(dN, dC, tr_radius) # (B,C,P,4,1)

  #       f_next = self._residual_ret(sp, o, vertices, smoothness, translations, var + tf.squeeze(dD, axis = -1), get_jac = False) # (B,C,P,4,1)

  #       actual_red = tf.expand_dims((tf.reduce_sum(f*f, axis = [-2, -1]) - tf.reduce_sum(f_next*f_next, axis = [-2, -1])), axis = -1) / 2.0 # (B,C,P,1)
  #       predict_red = -tf.matmul(tf.transpose(grad, [0, 1, 2, 4, 3]), dD) - tf.expand_dims(tf.expand_dims(tf.reduce_sum(tf.matmul(jac, dD)*tf.matmul(jac, dD), axis = [-2, -1]), axis = -1), axis = -1) / 2.0
  #       predict_red = tf.squeeze(predict_red, axis = -1) # (B,C,P,1)

  #       red_filter = tf.logical_and(predict_red < 1e-20, predict_red > -1e-20) # (B,C,P,1)
  #       # rho = tf.clip_by_value(actual_red / tf.where(tf.abs(predict_red) < 1e-10, 1e-10*tf.ones(tf.shape(predict_red)), tf.abs(predict_red)), clip_value_min = self._lb, clip_value_max = self._ub)
  #       rho = actual_red / tf.abs(predict_red)

  #       act_var = tf.expand_dims(tf.tile(actual_red, [1, 1, 1, 4]), axis = -1)
  #       pred_var = tf.expand_dims(tf.tile(predict_red, [1, 1, 1, 4]), axis = -1)

  #       large_rho = 1e+10 * tf.ones(tf.shape(rho))
  #       rho = tf.tile(tf.expand_dims(tf.where(red_filter, large_rho, rho), axis = -1), [1, 1, 1, 4, 1]) # (B,C,P,4,1)

  #       tr_filter1 = rho < 0.001
  #       tr_filter2 = rho > 0.9
  #       dD_norm = tf.tile(tf.linalg.norm(dD, axis = [-2, -1], keepdims = True), [1, 1, 1, 4, 1]) # (B,C,P,4,1)

  #       # tr_acc = (tr_acc + 1.0) * tf.cast(tr_filter1, tf.float32)
  #       # tr_acc = tf.clip_by_value(tr_acc, clip_value_min = 0.0, clip_value_max = 2.0)

  #       # tr_radius = tf.where(tr_filter1, tf.pow(0.1, tr_acc)*dD_norm, tr_radius)
  #       tr_radius = tf.where(tr_filter1, 0.25*dD_norm, tr_radius)
  #       tr_radius = tf.where(tr_filter2, tf.reduce_max(tf.concat([tr_radius, 3*dD_norm], axis = -1), axis = -1, keepdims = True), tr_radius) # (B,C,P,4,1)
  #       # tr_radius = tf.clip_by_value(tr_radius, clip_value_min = 1e-10, clip_value_max = 1000)

  #       var_filter = tf.squeeze(rho, axis = -1) > 0.001 # (B,C,P,4)
  #       var = tf.where(var_filter, var + tf.squeeze(dD, axis = -1), var) # (B,C,P,4)
  #       # var = tf.concat([var[:, :, :, :3] / self._safe_norm(var[:, :, :, :3], axis = -1, keepdims = True), var[:, :, :, 3:4]], axis = -1)

  #     else:
  #       pass
      
  #     # s = self._compute_spt(u_ie, vertices, smoothness, translations, get_h=False) # (B,C,P,3)
  #     # s_var = tf.expand_dims(tf.concat([s, tf.zeros([tf.shape(vertices)[0], n_c, tf.shape(points)[1], 1])], axis = -1), axis = -1) # (B,C,P,4,1)
  #     # tr_list.append(tf.expand_dims(tf.concat([tr_radius, dD, jac, jac_inv, f, dN, dC, tf.expand_dims(var, axis = -1), tf.tile(tf.expand_dims(tf.expand_dims(res, axis = -1), axis = -1), [1, 1, 1, 4, 1]), act_var, pred_var, initial_u], axis = -1), axis = -1)) # (B,C,P,4,18,1)


  #   "Record Growth Ratio"
  #   f = self._residual_ret(sp, o, vertices, smoothness, translations, var, get_jac = False) # (B,C,P,4,1)

  #   u_ie = tf.expand_dims(var[:, :, :, :3], axis = 3) # (B,C,P,1,3)
  #   u_ie = u_ie / tf.linalg.norm(u_ie, axis = [-2, -1], keepdims = True)
  #   u_ie = tf.squeeze(u_ie, axis = 3) # (B,C,P,3)
  #   u_ie = tf.stop_gradient(u_ie)
  #   s = self._compute_spt(u_ie, vertices, smoothness, translations, get_h=False) # (B,C,P,3)
  #   s = tf.expand_dims(s, axis = 3) # (B,C,P,1,3)

  #   ret = tf.reduce_min(self._safe_norm(sp - s, axis = -1), axis = 1) # (B,P,1)

  #   sigma = tf.clip_by_value(self._safe_norm(o_bar, axis = -1) / self._safe_norm(s - o, axis = -1), clip_value_min = 0.0, clip_value_max = self._ub_exponent)
  #   sigma = tf.reduce_min(sigma, axis = 1) # (B,P,1)

  #   loop_iter = tf.concat([[loop_iter], [g_res_iter], [g_iter]], axis = -1)
  #   # loop_iter = g_iter


  #   "End of Loop"
  #   retraction = ret

  #   undef_ret2 = tf.concat(tr_list, axis = -1)
  #   undef_ret = tf.concat([tf.squeeze(tf.concat([sp, s], axis = -1), axis = -2), u_ie], axis = -1) # (B,C,P,9)
  #   # undef_ret = res

  #   return retraction, loop_iter, undef_ret, undef_ret2

  def _compute_distance(self, vertices, smoothness, translations, points, get_points = False, get_normal = False, normal_filter = None):
    """Compute retarction state.

    Args:
      vertices: Tensor, [batch_size, n_parts, n_vertices, 3], convex local vertices.
      smoothness: Tensor, [batch_size, n_parts], convex smoothness.
      translations: Tensor, [batch_size, n_parts, 1, 3], convex centers.
      points: Tensor, [batch_size, n_parts, n_points, 3], point clouds.

    Returns:
      distance: Tensor, [batch_size, n_parts, n_points, 1], minimum distance for each point.
    """
    n_c = self._n_parts

    # undef_ret = tf.constant([[1.0], [1.0]])
    # loop_iter = tf.constant([0])
    # undef_list = []

    sp = tf.expand_dims(points, axis = 3) # (B,C,P,1,3)
    o = tf.tile(tf.expand_dims(translations, axis = 2), [1, 1, tf.shape(points)[2], 1, 1]) # (B,C,P,1,3)

    x_fw = tf.squeeze(o, axis = 3) # (B,C,P,3)
    u_fw = tf.squeeze(sp, axis = 3) - x_fw # (B,C,P,3)
    u_fw = u_fw / self._safe_norm(u_fw, axis = -1, keepdims = True)
    # undef_list.append(tf.expand_dims(u_fw, axis = -1))

    min_u = u_fw
    min_gap = 100*tf.ones_like(u_fw)
    gap = tf.zeros_like(u_fw)


    "FW Initialization"
    # iter = 0
    for j in range(20):
      # iter += 1
      
      s_fw = self._compute_spt(u_fw, vertices, smoothness, translations, get_h = False) # (B,C,P,3)

      gap = tf.tile(tf.linalg.norm((tf.squeeze(sp, axis = -2) - s_fw), axis = -1, keepdims = True), [1, 1, 1, 3])
      gap_filter = gap < min_gap
      min_u = tf.where(gap_filter, u_fw, min_u)
      min_gap = tf.where(gap_filter, gap, min_gap)

      prev_x = x_fw
      x_fw = x_fw + tf.squeeze(tf.matmul(tf.expand_dims(s_fw - x_fw, axis = -2), tf.expand_dims(tf.squeeze(sp, axis = -2) - x_fw, axis = -1)), axis = -1) / tf.linalg.norm((s_fw - x_fw), axis = -1, keepdims = True)**2 * (s_fw - x_fw)

      x_finite = tf.is_finite(s_fw - x_fw)
      x_filter = tf.tile(tf.linalg.norm(s_fw - x_fw, axis = -1, keepdims = True), [1, 1, 1, 3]) < 1e-30
      x_fw = tf.where(tf.logical_or(x_filter, tf.logical_not(x_finite)), prev_x, x_fw)

      prev_u = u_fw
      u_fw = tf.squeeze(sp, axis = -2) - x_fw
      u_finite = tf.is_finite(u_fw)
      u_filter = tf.tile(tf.linalg.norm(u_fw, axis = -1, keepdims = True), [1, 1, 1, 3]) < 1e-30
      u_fw = tf.where(tf.logical_or(u_filter, tf.logical_not(u_finite)), prev_u, u_fw)

      u_fw = u_fw / tf.linalg.norm(u_fw, axis = -1, keepdims = True)
      # undef_list.append(tf.expand_dims(u_fw, axis = -1))
    
    # undef_ret = tf.concat(undef_list, axis = -1) # (B,C,P,3,31)
    # loop_iter = iter

    # init_u = u_fw
    s = self._compute_spt(u_fw, vertices, smoothness, translations, get_h=False) # (B,C,P,3)

    gap = tf.tile(tf.linalg.norm((tf.squeeze(sp, axis = -2) - s), axis = -1, keepdims = True), [1, 1, 1, 3])
    gap_filter = gap < min_gap
    u_fw = tf.where(gap_filter, u_fw, min_u)
    
    s = tf.expand_dims(s, axis = 3) # (B,C,P,1,3)


    "Growth Model"
    sig = tf.clip_by_value(tf.squeeze(tf.matmul(sp - s, tf.expand_dims(u_fw, axis = -1)), axis = -1), clip_value_min = self._lb, clip_value_max = self._ub) # (B,C,P,1)
    var = tf.concat([u_fw, sig], axis = -1) # (B,C,P,4)

    tr_radius = 10.0 * tf.ones((tf.shape(vertices)[0], n_c, tf.shape(points)[2], 4, 1)) # (B,C,P,4,1)
    # g_iter = 0

    for j in range(10):
      f, jac = self._residual_distance(sp, o, vertices, smoothness, translations, var) # (B,C,P,4,1), (B,C,P,4,4)
      res = tf.linalg.norm(f, axis = [-2, -1]) # (B,C,P)

      # jac_det = tf.linalg.det(jac) # (B,C,P)
      # jac_det_min = tf.reduce_min(tf.abs(jac_det))
      # jac_det_max = tf.reduce_max(tf.abs(jac_det))
      # jac_finite = tf.reduce_min(tf.cast(tf.is_finite(jac), tf.float32))

      # jac_finite_filter = tf.is_finite(tf.tile(tf.expand_dims(jac_det, axis = -1), [1, 1, 1, 4]))

      dD = tf.ones_like(tr_radius)
      dD_norm = tf.ones_like(dD)
      dN = tf.ones_like(tr_radius)
      dC = tf.ones_like(tr_radius)
      jac_inv = tf.ones_like(jac)
      
      if tf.reduce_max(res) > 1e-6:
        # g_iter += 1
      
        # jac_inv = tf.linalg.inv(jac)
        jac_inv = self._cramer_inv(jac, 4) # (B,C,P,4,4)

        dN = -tf.matmul(jac_inv, f) # (B,C,P,4,1)
        grad = tf.matmul(tf.transpose(jac, [0, 1, 2, 4, 3]), f) # (B,C,P,4,1)
        dC = - tf.reduce_sum(grad*grad, axis = [-2, -1], keepdims = True) / (tf.reduce_sum(tf.matmul(jac, grad)*tf.matmul(jac, grad), axis = [-2, -1], keepdims = True) + 1e-30) * grad
        dD = self._dogleg_ret(dN, dC, tr_radius) # (B,C,P,4,1)

        f_next = self._residual_distance(sp, o, vertices, smoothness, translations, var + tf.squeeze(dD, axis = -1), get_jac = False) # (B,C,P,4,1)

        actual_red = tf.expand_dims((tf.reduce_sum(f*f, axis = [-2, -1]) - tf.reduce_sum(f_next*f_next, axis = [-2, -1])), axis = -1) / 2.0 # (B,C,P,1)
        predict_red = -tf.matmul(tf.transpose(grad, [0, 1, 2, 4, 3]), dD) - tf.expand_dims(tf.expand_dims(tf.reduce_sum(tf.matmul(jac, dD)*tf.matmul(jac, dD), axis = [-2, -1]), axis = -1), axis = -1) / 2.0
        predict_red = tf.squeeze(predict_red, axis = -1) # (B,C,P,1)

        red_filter = tf.logical_and(predict_red < 1e-20, predict_red > -1e-20) # (B,C,P,1)
        # rho = tf.clip_by_value(actual_red / tf.where(tf.abs(predict_red) < 1e-10, 1e-10*tf.ones(tf.shape(predict_red)), tf.abs(predict_red)), clip_value_min = self._lb, clip_value_max = self._ub)
        rho = actual_red / predict_red

        large_rho = 1e+10 * tf.ones(tf.shape(rho))
        rho = tf.tile(tf.expand_dims(tf.where(red_filter, large_rho, rho), axis = -1), [1, 1, 1, 4, 1]) # (B,C,P,4,1)

        tr_filter1 = rho < 0.05
        tr_filter2 = rho > 0.9
        dD_norm = tf.tile(tf.linalg.norm(dD, axis = [-2, -1], keepdims = True), [1, 1, 1, 4, 1]) # (B,C,P,4,1)

        # tr_acc = (tr_acc + 1.0) * tf.cast(tr_filter1, tf.float32)
        # tr_acc = tf.clip_by_value(tr_acc, clip_value_min = 0.0, clip_value_max = 2.0)

        # tr_radius = tf.where(tr_filter1, tf.pow(0.1, tr_acc)*dD_norm, tr_radius)
        tr_radius = tf.where(tr_filter1, 0.25*dD_norm, tr_radius)
        tr_radius = tf.where(tr_filter2, tf.reduce_max(tf.concat([tr_radius, 3*dD_norm], axis = -1), axis = -1, keepdims = True), tr_radius) # (B,C,P,4,1)
        # tr_radius = tf.clip_by_value(tr_radius, clip_value_min = 1e-10, clip_value_max = 1000)

        var_filter = tf.squeeze(rho, axis = -1) > 0.05 # (B,C,P,4)
        var = tf.where(var_filter, var + tf.squeeze(dD, axis = -1), var) # (B,C,P,4)
        # var = tf.concat([var[:, :, :, :3] / self._safe_norm(var[:, :, :, :3], axis = -1, keepdims = True), var[:, :, :, 3:4]], axis = -1)

      else:
        pass
      
      # s = self._compute_spt(u_ie, vertices, smoothness, translations, get_h=False) # (B,C,P,3)
      # s_var = tf.expand_dims(tf.concat([s, tf.zeros([tf.shape(vertices)[0], n_c, tf.shape(points)[1], 1])], axis = -1), axis = -1) # (B,C,P,4,1)
      # tr_list.append(tf.expand_dims(tf.concat([tr_radius, dD, jac, jac_inv, f, dN, dC, tf.expand_dims(var, axis = -1), tf.tile(tf.expand_dims(tf.expand_dims(res, axis = -1), axis = -1), [1, 1, 1, 4, 1]), act_var, pred_var, initial_u], axis = -1), axis = -1)) # (B,C,P,4,18,1)


    "Record Distances"
    f = self._residual_distance(sp, o, vertices, smoothness, translations, var, get_jac = False) # (B,C,P,4,1)

    u = tf.expand_dims(var[:, :, :, :3], axis = 3) # (B,C,P,1,3)
    u = u / tf.linalg.norm(u, axis = [-2, -1], keepdims = True)
    u = tf.squeeze(u, axis = 3) # (B,C,P,3)
    u = tf.stop_gradient(u)
    s = self._compute_spt(u, vertices, smoothness, translations, get_h=False) # (B,C,P,3)
    s = tf.expand_dims(s, axis = 3) # (B,C,P,1,3)

    # dist = self._safe_norm(sp - s, axis = -1) # (B,C,P,1)
    # sgn = tf.cast(tf.squeeze(tf.matmul((sp - s), tf.expand_dims(u_ie, axis = -1)), axis = -1) >= 0, tf.float32) # (B,C,P,1)
    # sgn = 2*sgn - 1

    # loop_iter = tf.concat([[loop_iter], [g_iter]], axis = -1)
    # loop_iter = g_iter


    "End of Loop"
    distance = tf.squeeze(tf.matmul((sp - s), tf.expand_dims(u, axis = -1)), axis = -1) # (B,C,P,1)

    # undef_ret = tf.concat([undef_ret, tf.transpose(sp, [0, 1, 2, 4, 3]), tf.transpose(s, [0, 1, 2, 4, 3]), tf.expand_dims(u, axis = -1), tf.expand_dims(init_u, axis = -1)], axis = -1) # (B,C,P,3,31+4)
    # # undef_ret = res

    if not get_points:
      # return distance, loop_iter, undef_ret
      if normal_filter is not None:
        return tf.where(normal_filter, 10.0*tf.ones_like(distance), distance) # (B,C,P,1)
      else:
        return distance
    else:
      if not get_normal:
        return tf.squeeze(s, axis = 3) # (B,C,P,3)
      else:
        return tf.squeeze(s, axis = 3), u # (B,C,P,3), (B,C,P,3)

  # def _residual(self, o_bar, i_v, i_p, i_t, vertices, smoothness, translations, var, get_jac = True):
  #   x = tf.expand_dims(var[:, :, :3], axis = 2)

  #   s1, dsdx1 = self._compute_spt(x, i_v, i_p, i_t, get_h=False, get_dsdx=True)
  #   s2, dsdx2 = self._compute_spt(-x, vertices, smoothness, translations, get_h=False, get_dsdx=True)

  #   f = tf.concat([(var[:, :, 3:4]*tf.squeeze(s1 - s2, axis = 2) + (1 - var[:, :, 3:4])*tf.squeeze(-o_bar, axis = 2)), (tf.expand_dims(tf.reduce_sum(x*x, axis = [2, 3]) - 1.0, axis = -1))], axis = -1)
  #   f = tf.expand_dims(f, axis = -1)

  #   if get_jac:
  #     jac = tf.expand_dims(var[:, :, 3:4], axis = -1) * tf.squeeze(dsdx1 + dsdx2, axis = 2)
  #     jac = tf.concat([jac, tf.transpose(s1 - s2 + o_bar, [0, 1, 3, 2])], axis = -1)
  #     jac = tf.concat([jac, tf.concat([2.0*x, tf.zeros((tf.shape(vertices)[0], tf.shape(vertices)[1], 1, 1))], axis = -1)], axis = -2)
  #     jac = tf.clip_by_value(jac, clip_value_min = -1e-30, clip_value_max = 1e+30)
  #     return f, jac
  #   else:
  #     return f
  
  def _residual(self, o_bar, vertices, smoothness, translations, var, get_jac = True):
    x = var[:, :, :, :3] # (B,C,C,3)

    s1, dsdx1 = self._compute_spt(x, vertices, smoothness, translations, get_h=False, get_dsdx=True)
    s2, dsdx2 = self._compute_spt(tf.transpose(-x, [0, 2, 1, 3]), vertices, smoothness, translations, get_h=False, get_dsdx=True)
    s2 = tf.transpose(s2, [0, 2, 1, 3]) # (B,C,C,3)
    dsdx2 = tf.transpose(dsdx2, [0, 2, 1, 3, 4]) # (B,C,C,3,3)

    f = tf.concat([(var[:, :, :, 3:4]*(s1 - s2) + (1 - var[:, :, :, 3:4])*tf.squeeze(-o_bar, axis = -1)), (tf.reduce_sum(x*x, axis = -1, keepdims = True) - 1.0)], axis = -1)
    f = tf.expand_dims(f, axis = -1) # (B,C,C,4,1)

    if get_jac:
      jac = tf.expand_dims(var[:, :, :, 3:4], axis = -1) * (dsdx1 + dsdx2)
      jac = tf.concat([jac, tf.expand_dims(s1 - s2, axis = -1) + o_bar], axis = -1) # (B,C,C,3,4)
      jac = tf.concat([jac, tf.concat([2.0*tf.expand_dims(x, axis = -2), tf.zeros((tf.shape(vertices)[0], tf.shape(vertices)[1], tf.shape(vertices)[1], 1, 1))], axis = -1)], axis = -2)
      jac = tf.clip_by_value(jac, clip_value_min = -1e+30, clip_value_max = 1e+30) # (B,C,C,4,4)
      return f, jac
    else:
      return f

  def _residual_ret(self, sp, o, vertices, smoothness, translations, var, get_jac = True):
    x = tf.expand_dims(var[:, :, :, :3], axis = 3) # (B,C,P,1,3)
    xd = var[:, :, :, :3] # (B,C,P,3)

    s, dsdx = self._compute_spt(xd, vertices, smoothness, translations, get_h = False, get_dsdx = True) # (B,C,P,3), (B,C,P,3,3)
    s = tf.expand_dims(s, axis = 3) # (B,C,P,1,3)
    dsdx = tf.expand_dims(dsdx, axis = 3) # (B,C,P,1,3,3)

    f = tf.concat([(var[:, :, :, 3:4]*tf.squeeze(s - sp, axis = 3) + (1 - var[:, :, :, 3:4])*tf.squeeze(o - sp, axis = 3)), (tf.expand_dims(tf.reduce_sum(x*x, axis = [-2, -1]) - 1.0, axis = -1))], axis = -1)
    f = tf.expand_dims(f, axis = -1) # (B,C,P,4,1)

    if get_jac:
      jac = tf.expand_dims(var[:, :, :, 3:4], axis = -1) * tf.squeeze(dsdx, axis = 3) # (B,C,P,3,3)
      jac = tf.concat([jac, tf.transpose(s - o, [0, 1, 2, 4, 3])], axis = -1) # (B,C,P,3,4)
      jac = tf.concat([jac, tf.concat([2.0*x, tf.zeros((tf.shape(vertices)[0], self._n_parts, tf.shape(sp)[2], 1, 1))], axis = -1)], axis = -2) # (B,C,P,4,4)
      jac = tf.clip_by_value(jac, clip_value_min = -self._ub, clip_value_max = self._ub)
      return f, jac
    else:
      return f
  
  def _residual_distance(self, sp, o, vertices, smoothness, translations, var, get_jac = True):
    x = tf.expand_dims(var[:, :, :, :3], axis = 3) # (B,C,P,1,3)
    xd = var[:, :, :, :3] # (B,C,P,3)

    s, dsdx = self._compute_spt(xd, vertices, smoothness, translations, get_h = False, get_dsdx = True) # (B,C,P,3), (B,C,P,3,3)
    s = tf.expand_dims(s, axis = 3) # (B,C,P,1,3)
    dsdx = tf.expand_dims(dsdx, axis = 3) # (B,C,P,1,3,3)

    f = tf.concat([(tf.squeeze(sp - s, axis = 3) - var[:, :, :, 3:4]*xd), (tf.expand_dims(tf.reduce_sum(x*x, axis = [-2, -1]) - 1.0, axis = -1))], axis = -1)
    f = tf.expand_dims(f, axis = -1) # (B,C,P,4,1)

    if get_jac:
      jac = -tf.squeeze(dsdx, axis = 3) - tf.expand_dims(var[:, :, :, 3:4], axis = -1)*tf.reshape(tf.eye(3), [1, 1, 1, 3 ,3]) # (B,C,P,3,3)
      jac = tf.concat([jac, -tf.transpose(x, [0, 1, 2, 4, 3])], axis = -1) # (B,C,P,3,4)
      jac = tf.concat([jac, tf.concat([2.0*x, tf.zeros((tf.shape(vertices)[0], self._n_parts, tf.shape(sp)[2], 1, 1))], axis = -1)], axis = -2) # (B,C,P,4,4)
      jac = tf.clip_by_value(jac, clip_value_min = -1e+30, clip_value_max = 1e+30)
      return f, jac
    else:
      return f

  def _dogleg(self, dN, dC, tr_radius):
    dN_norm = tf.tile(tf.linalg.norm(dN, axis = [-2, -1], keepdims = True), [1, 1, 4, 1])
    dC_norm = tf.tile(tf.linalg.norm(dC, axis = [-2, -1], keepdims = True), [1, 1, 4, 1])
    dN_filter = dN_norm < tr_radius
    dC_filter = dC_norm > tr_radius

    a = tf.tile(tf.linalg.norm(dN - dC, axis = [-2, -1], keepdims = True), [1, 1, 4, 1])
    a = a*a
    a_filter = a > 1e+30
    a = tf.clip_by_value(a, clip_value_min = 1e-20, clip_value_max = 1e+20)
    b = tf.matmul(tf.transpose(dC, [0, 1, 3, 2]), dN - dC)
    c = b*b - a*(tf.reduce_sum(dC*dC, axis = [-2, -1], keepdims = True) - tr_radius*tr_radius)
    tau = (-b + tf.sqrt(tf.abs(c))) / a
    tau = tf.where(a_filter, tf.zeros_like(tau), tau)

    dD = tf.where(dN_filter, dN, tf.zeros_like(dN))

    dD = tf.where(tf.logical_and(tf.logical_not(dN_filter), dC_filter), tr_radius*dC / self._safe_norm(dC, axis = [-2, -1], keepdims = True), dD)
    dD = tf.where(tf.logical_and(tf.logical_not(dN_filter), tf.logical_not(dC_filter)), (dC + tau*(dN - dC)), dD)

    return dD
  
  def _dogleg_ret(self, dN, dC, tr_radius):
    dN_norm = tf.tile(tf.linalg.norm(dN, axis = [-2, -1], keepdims = True), [1, 1, 1, 4, 1]) # (B,C,P,4,1)
    dC_norm = tf.tile(tf.linalg.norm(dC, axis = [-2, -1], keepdims = True), [1, 1, 1, 4, 1]) # (B,C,P,4,1)
    dN_filter = dN_norm < tr_radius # (B,C,P,4,1)
    dC_filter = dC_norm > tr_radius # (B,C,P,4,1)

    a = tf.tile(tf.linalg.norm(dN - dC, axis = [-2, -1], keepdims = True), [1, 1, 1, 4, 1]) # (B,C,P,4,1)
    a = a*a
    a_filter = a > 1e+30
    a = tf.clip_by_value(a, clip_value_min = 1e-30, clip_value_max = 1e+30) # (B,C,P,4,1)
    b = tf.matmul(tf.transpose(dC, [0, 1, 2, 4, 3]), dN - dC) # (B,C,P,1,1)
    c = b*b - a*(tf.reduce_sum(dC*dC, axis = [-2, -1], keepdims = True) - tr_radius*tr_radius) # (B,C,P,4,1)
    tau = (-b + tf.sqrt(tf.abs(c))) / a # (B,C,P,4,1)
    tau = tf.where(a_filter, tf.zeros_like(tau), tau)

    dD = tf.where(dN_filter, dN, tf.zeros_like(dN))

    dD = tf.where(tf.logical_and(tf.logical_not(dN_filter), dC_filter), tr_radius*dC / self._safe_norm(dC, axis = [-2, -1], keepdims = True), dD)
    dD = tf.where(tf.logical_and(tf.logical_not(dN_filter), tf.logical_not(dC_filter)), (dC + tau*(dN - dC)), dD)

    return dD # (B,C,P,4,1)

  def _safe_norm(self, x, axis, keepdims = False):
    norm = tf.sqrt(tf.reduce_sum(x*x, axis = axis, keepdims = keepdims) + 1e-30)
    return norm
  
  def _cramer_inv(self, x, dims):
    detx = tf.expand_dims(tf.expand_dims(tf.linalg.det(x), axis = -1), axis = -1)

    if dims == 3:
      c11 = tf.expand_dims(tf.expand_dims(tf.linalg.det(x[..., 1:3, 1:3]), axis = -1), axis = -1)
      c12 = -tf.expand_dims(tf.expand_dims(tf.linalg.det(tf.concat([x[..., 1:3, 0:1], x[..., 1:3, 2:3]], axis = -1)), axis = -1), axis = -1)
      c13 = tf.expand_dims(tf.expand_dims(tf.linalg.det(x[..., 1:3, 0:2]), axis = -1), axis = -1)

      c21 = -tf.expand_dims(tf.expand_dims(tf.linalg.det(tf.concat([x[..., 0:1, 1:3], x[..., 2:3, 1:3]], axis = -2)), axis = -1), axis = -1)
      c22 = tf.expand_dims(tf.expand_dims(tf.linalg.det(tf.concat([tf.concat([x[..., 0:1, 0:1], x[..., 0:1, 2:3]], axis = -1), tf.concat([x[..., 2:3, 0:1], x[..., 2:3, 2:3]], axis = -1)], axis = -2)), axis = -1), axis = -1)
      c23 = -tf.expand_dims(tf.expand_dims(tf.linalg.det(tf.concat([x[..., 0:1, 0:2], x[..., 2:3, 0:2]], axis = -2)), axis = -1), axis = -1)

      c31 = tf.expand_dims(tf.expand_dims(tf.linalg.det(x[..., 0:2, 1:3]), axis = -1), axis = -1)
      c32 = -tf.expand_dims(tf.expand_dims(tf.linalg.det(tf.concat([x[..., 0:2, 0:1], x[..., 0:2, 2:3]], axis = -1)), axis = -1), axis = -1)
      c33 = tf.expand_dims(tf.expand_dims(tf.linalg.det(x[..., 0:2, 0:2]), axis = -1), axis = -1)

      inv_1 = tf.concat([c11, c21, c31], axis = -1)
      inv_2 = tf.concat([c12, c22, c32], axis = -1)
      inv_3 = tf.concat([c13, c23, c33], axis = -1)
      invx = tf.concat([inv_1, inv_2, inv_3], axis = -2) / detx

      detx = tf.concat([detx, detx, detx], axis = -2)
      detx = tf.concat([detx, detx, detx], axis = -1)
      det_filter = tf.abs(detx) < 1e-30
      invx = tf.where(det_filter, tf.ones(tf.shape(invx))*1e+20, invx)
      return invx
    
    elif dims == 4:
      c11 = tf.expand_dims(tf.expand_dims(tf.linalg.det(x[..., 1:4, 1:4]), axis = -1), axis = -1)
      c12 = -tf.expand_dims(tf.expand_dims(tf.linalg.det(tf.concat([x[..., 1:4, 0:1], x[..., 1:4, 2:4]], axis = -1)), axis = -1), axis = -1)
      c13 = tf.expand_dims(tf.expand_dims(tf.linalg.det(tf.concat([x[..., 1:4, 0:2], x[..., 1:4, 3:4]], axis = -1)), axis = -1), axis = -1)
      c14 = -tf.expand_dims(tf.expand_dims(tf.linalg.det(x[..., 1:4, 0:3]), axis = -1), axis = -1)

      c21 = -tf.expand_dims(tf.expand_dims(tf.linalg.det(tf.concat([x[..., 0:1, 1:4], x[..., 2:4, 1:4]], axis = -2)), axis = -1), axis = -1)
      c22 = tf.expand_dims(tf.expand_dims(tf.linalg.det(tf.concat([tf.concat([x[..., 0:1, 0:1], x[..., 0:1, 2:4]], axis = -1), tf.concat([x[..., 2:4, 0:1], x[..., 2:4, 2:4]], axis = -1)], axis = -2)), axis = -1), axis = -1)
      c23 = -tf.expand_dims(tf.expand_dims(tf.linalg.det(tf.concat([tf.concat([x[..., 0:1, 0:2], x[..., 0:1, 3:4]], axis = -1), tf.concat([x[..., 2:4, 0:2], x[..., 2:4, 3:4]], axis = -1)], axis = -2)), axis = -1), axis = -1)
      c24 = tf.expand_dims(tf.expand_dims(tf.linalg.det(tf.concat([x[..., 0:1, 0:3], x[..., 2:4, 0:3]], axis = -2)), axis = -1), axis = -1)

      c31 = tf.expand_dims(tf.expand_dims(tf.linalg.det(tf.concat([x[..., 0:2, 1:4], x[..., 3:4, 1:4]], axis = -2)), axis = -1), axis = -1)
      c32 = -tf.expand_dims(tf.expand_dims(tf.linalg.det(tf.concat([tf.concat([x[..., 0:2, 0:1], x[..., 0:2, 2:4]], axis = -1), tf.concat([x[..., 3:4, 0:1], x[..., 3:4, 2:4]], axis = -1)], axis = -2)), axis = -1), axis = -1)
      c33 = tf.expand_dims(tf.expand_dims(tf.linalg.det(tf.concat([tf.concat([x[..., 0:2, 0:2], x[..., 0:2, 3:4]], axis = -1), tf.concat([x[..., 3:4, 0:2], x[..., 3:4, 3:4]], axis = -1)], axis = -2)), axis = -1), axis = -1)
      c34 = -tf.expand_dims(tf.expand_dims(tf.linalg.det(tf.concat([x[..., 0:2, 0:3], x[..., 3:4, 0:3]], axis = -2)), axis = -1), axis = -1)

      c41 = -tf.expand_dims(tf.expand_dims(tf.linalg.det(x[..., 0:3, 1:4]), axis = -1), axis = -1)
      c42 = tf.expand_dims(tf.expand_dims(tf.linalg.det(tf.concat([x[..., 0:3, 0:1], x[..., 0:3, 2:4]], axis = -1)), axis = -1), axis = -1)
      c43 = -tf.expand_dims(tf.expand_dims(tf.linalg.det(tf.concat([x[..., 0:3, 0:2], x[..., 0:3, 3:4]], axis = -1)), axis = -1), axis = -1)
      c44 = tf.expand_dims(tf.expand_dims(tf.linalg.det(x[..., 0:3, 0:3]), axis = -1), axis = -1)

      inv_1 = tf.concat([c11, c21, c31, c41], axis = -1)
      inv_2 = tf.concat([c12, c22, c32, c42], axis = -1)
      inv_3 = tf.concat([c13, c23, c33, c43], axis = -1)
      inv_4 = tf.concat([c14, c24, c34, c44], axis = -1)
      invx = tf.concat([inv_1, inv_2, inv_3, inv_4], axis = -2) / detx

      detx = tf.concat([detx, detx, detx, detx], axis = -2)
      detx = tf.concat([detx, detx, detx, detx], axis = -1)
      det_filter = tf.abs(detx) < 1e-30
      invx = tf.where(det_filter, tf.ones(tf.shape(invx))*1e+20, invx)
      return invx
