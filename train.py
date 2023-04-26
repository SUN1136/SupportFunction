from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow as tf2

from os import path

from lib import datasets
from lib import models
from lib import utils

tf.disable_eager_execution()

gpus = tf2.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # memory limit 10 times increased
        tf2.config.experimental.set_virtual_device_configuration(gpus[0], [tf2.config.experimental.VirtualDeviceConfiguration(memory_limit=1500000)])
        # tf.config.experimental.set_memory_growth(gpus[0], True)
        print("\n----------GPU Loaded----------\n")
    except RuntimeError as e:
        print(e)

logging = tf.logging
tf.logging.set_verbosity(tf.logging.INFO)

flags = utils.define_flags()
FLAGS = flags.FLAGS

def main(unused_argv):
  tf.set_random_seed(454545)
  np.random.seed(454545)

  logging.info("=> Starting ...")

  # Select dataset.
  logging.info("=> Preparing datasets ...")
  data = datasets.get_dataset(FLAGS.dataset, "train", FLAGS)
  batch = tf.data.make_one_shot_iterator(data).get_next()
  print(data)

  # Select model.
  logging.info("=> Creating {} model".format(FLAGS.model))
  model = models.get_model(FLAGS.model, FLAGS)
  # optimizer = tf.train.AdamOptimizer(FLAGS.lr)

  # Set up the graph
  train_loss, train_op, global_step, out_points, beta, vert, smooth, direc, locvert, dhdz, zm, points, overlap, iter, undef = model.compute_loss(
      batch, training=True, optimizer=tf.train.AdamOptimizer)

  # Training hooks
  stop_hook = tf.train.StopAtStepHook(last_step=FLAGS.max_steps)
  summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
  ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
  summary_hook = tf.train.SummarySaverHook(
      save_steps=1, summary_writer=summary_writer, summary_op=ops)
  step_counter_hook = tf.train.StepCounterHook(summary_writer=summary_writer)
  hooks = [stop_hook, step_counter_hook, summary_hook]

  sing_list = []
  sing = 0
  logging.info("=> Start training loop ...")
  log_count = 1000
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.train_dir,
      hooks=hooks,
      scaffold=None,
      save_checkpoint_steps=FLAGS.save_every,
      save_checkpoint_secs=None,
      save_summaries_steps=None,
      save_summaries_secs=None,
      log_step_count_steps=log_count,
      max_wait_secs=3600) as mon_sess:

    while not mon_sess.should_stop():
      unused_var, loss_var, step_var, unused_var4, out_var, beta_var, vert_var, smooth_var, direc_var, locvert_var, dhdz_var, zm_var, points_var, overlap_var, iter_var, undef_var = mon_sess.run(
        [batch, train_loss, global_step, train_op, out_points, beta, vert, smooth, direc, locvert, dhdz, zm, points, overlap, iter, undef])
      if step_var % 1000 == 0:
        print("")
        print("Step: ", step_var, "\t\tLoss: ", loss_var)
        print("Smoothness: ", smooth_var[0, :])
        print("Growth Ratio: ", overlap_var[0, :, 0])
        print("Loop Iter: ", iter_var)
        print("Undef: ", undef_var)
        print("")
        sing_list.append(sing)
        sing = 0
      if undef_var[0, 0, -1, 0] != 0:
        print("")
        print("----------Singularity----------")
        print("Step: ", step_var, "\t\tLoss: ", loss_var)
        print("Smoothness: ", smooth_var[0, :])
        print("Growth Ratio: ", overlap_var[0, :, 0])
        print("Loop Iter: ", iter_var)
        print("Undef: ", undef_var)
        print("")
        sing += 1
        # raise Exception("----------Singularity----------")
      if step_var >= FLAGS.max_steps - 1:
        sing_list = np.array(sing_list)
        print("")
        print("Singularity result: ", sing_list)
        print("")
        
        out_var = out_var[0, :, :]
        smooth_var = smooth_var[0, :]
        # beta_var = beta_var[0, :].reshape(-1, )
        vert_var = vert_var[0, ...].reshape(-1, 3)
        # locvert_var = locvert_var[0, ...].reshape(-1, 3)
        # dhdz_var = dhdz_var[0, ...].reshape(-1, dhdz_var.shape[-1])
        # zm_var = zm_var[0, ...].reshape(-1, zm_var.shape[-1])
        points_var = points_var[0, ...]
        overlap_var = overlap_var[0, :, 0]
        with tf.io.gfile.GFile(path.join(FLAGS.train_dir, "stats.csv"), "w") as fout:
          fout.write("x,y,z\n")
          for i in range(out_var.shape[0]):
            fout.write("{0},{1},{2}\n".format(out_var[i, 0], out_var[i, 1], out_var[i, 2]))
          fout.write("p\n")
          for i in range(smooth_var.shape[0]):
            fout.write("{}\n".format(smooth_var[i]))
          # fout.write("beta\n")
          # for i in range(beta_var.shape[0]):
          #   fout.write("{}\n".format(beta_var[i]))
          fout.write("vertex\n")
          for i in range(vert_var.shape[0]):
            fout.write("{0},{1},{2}\n".format(vert_var[i, 0], vert_var[i, 1], vert_var[i, 2]))
          # fout.write("directions\n")
          # for i in range(direc_var.shape[0]):
          #   fout.write("{0},{1},{2}\n".format(direc_var[i, 0], direc_var[i, 1], direc_var[i, 2]))
          # fout.write("local vertex\n")
          # for i in range(locvert_var.shape[0]):
          #   fout.write("{0},{1},{2}\n".format(locvert_var[i, 0], locvert_var[i, 1], locvert_var[i, 2]))
          # fout.write("dhdz\n")
          # for i in range(dhdz_var.shape[0]):
          #   fout.write("{0},{1},{2}\n".format(dhdz_var[i, 0], dhdz_var[i, 1], dhdz_var[i, 2]))
          # fout.write("zm\n")
          # for i in range(zm_var.shape[0]):
          #   fout.write("{0},{1},{2}\n".format(zm_var[i, 0], zm_var[i, 1], zm_var[i, 2]))
          fout.write("dataset\n")
          for i in range(points_var.shape[0]):
            fout.write("{0},{1},{2}\n".format(points_var[i, 0], points_var[i, 1], points_var[i, 2]))
          fout.write("overlap\n")
          for i in range(overlap_var.shape[0]):
            fout.write("{}\n".format(overlap_var[i]))


if __name__ == "__main__":
  tf.app.run(main)

