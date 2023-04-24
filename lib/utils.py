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
"""Utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from os import path
import numpy as np
import tensorflow.compat.v1 as tf

from lib import datasets
from lib import models


def define_flags():
  """Define command line flags."""
  flags = tf.app.flags
  
  # Model flags
  flags.DEFINE_string("model", "multiconvex", "Name of the model.")
  flags.DEFINE_integer("n_parts", 2, "Number of convexes uesd.")
  flags.DEFINE_integer("n_vertices", 10, "Number of vertices uesd.")
  flags.DEFINE_integer("latent_size", 128, "The size of latent code.")
  # flags.DEFINE_integer("dims", 3, "The dimension of query points.")
  flags.DEFINE_bool("image_input", False, "Use images as input if True.")
  flags.DEFINE_float("vis_scale", 1.3,
                     "Scale of bbox used when extracting meshes.")

  # Dataset flags
  flags.DEFINE_string("dataset", "shapenet", "Name of the dataset.")
  flags.DEFINE_string("data_dir", None, "The base directory to load data from.")
  flags.mark_flag_as_required("data_dir")
  flags.DEFINE_string("obj_class", "*", "Object class used from dataset.")

  # Training flags
  flags.DEFINE_float("lr", 1e-3, "Start learning rate.")
  flags.DEFINE_string(
      "train_dir", None, "The base directory to save training info and"
      "checkpoints.")
  flags.DEFINE_integer("save_every", 20000,
                       "The number of steps to save checkpoint.")
  flags.DEFINE_integer("max_steps", 2000, "The number of steps of training.")
  flags.DEFINE_integer("batch_size", 1, "Batch size.")
  flags.DEFINE_integer("sample_point", 1024, "The number of sample points.")
  flags.DEFINE_integer("n_convex_altitude", 31, "The output surface resolution angle degrees.")
  # flags.DEFINE_integer("n_mesh_inter", 5, "The mesh interpolation factor.")
  # flags.DEFINE_integer("n_top_k", 50, "The number of meshes for top k sampling.")
  # flags.DEFINE_integer("n_bottom_k", 0, "The number of meshes for bottom k sampling.")
  # flags.DEFINE_bool("use_surface_sampling", False, "Use surface sampling points for training.")
  flags.mark_flag_as_required("train_dir")

  # Eval flags

  return flags
