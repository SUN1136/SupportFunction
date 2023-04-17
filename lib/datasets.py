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
"""Dataset implementations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path

import tensorflow.compat.v1 as tf


def get_dataset(data_name, split, args):
  return dataset_dict[data_name](split, args)


def shapenet(split, args):
  """Sample Point Cloud Dataset.

  Args:
    split: string, the split of the dataset, either "train" or "test".
    args: tf.app.flags.FLAGS, configurations.

  Returns:
    dataset: tf.data.Dataset, the point cloud dataset.
  """
  total_points = 100000
  data_dir = args.data_dir
  sample_point = args.sample_point
  batch_size = args.batch_size if split == "train" else 1
  dims = 3

  def _parser(example):
    fs = tf.parse_single_example(
        example,
        features={
            "point_samples":
                tf.FixedLenFeature([total_points * dims], tf.float32)
        })
    fs["point_samples"] = tf.reshape(fs["point_samples"], [total_points, dims])
    return fs

  def _sampler(example):

    points = []

    if sample_point > 0:
      if split == "train":
        indices_bbx = tf.random.uniform([sample_point],
                                        minval=0,
                                        maxval=total_points,
                                        dtype=tf.int32)
        point_samples = tf.gather(example["point_samples"], indices_bbx, axis=0)
      else:
        point_samples = example["point_samples"]
      points.append(point_samples)

    points = tf.reshape(points, [sample_point, 3])

    return {
        "point": points
    }

  data_pattern = path.join(data_dir, "{}-{}-*".format(args.obj_class, split))
  data_files = tf.gfile.Glob(data_pattern)
  if not data_files:
    raise ValueError("{} did not match any files".format(data_pattern))
  file_count = len(data_files)
  filenames = tf.data.Dataset.list_files(data_pattern, shuffle=True)
  data = filenames.interleave(
      lambda x: tf.data.TFRecordDataset([x]),
      cycle_length=file_count,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
  data = data.map(_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  data = data.map(_sampler, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if split == "train":
    data = data.shuffle(batch_size * 5).repeat(-1)

  return data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


dataset_dict = {
    "shapenet": shapenet,
}
