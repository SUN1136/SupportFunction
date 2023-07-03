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

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # memory limit 10 times increased
        # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        # logical_gpus = tf2.config.experimental.list_logical_devices('GPU')
        print("\n----------GPU Loaded----------\n")
    except RuntimeError as e:
        print(e)

logging = tf.logging
tf.logging.set_verbosity(tf.logging.INFO)

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.8

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
  # train_loss, train_op, global_step, out_points, vert, smooth, points, overlap = model.compute_loss(
  #     batch, training=True, optimizer=tf.train.AdamOptimizer)
  train_loss, train_op, global_step, vert, smooth, overlap = model.compute_loss(
      batch, training=True, optimizer=tf.train.AdamOptimizer)

  # Training hooks
  stop_hook = tf.train.StopAtStepHook(last_step=FLAGS.max_steps)
  summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
  ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
  summary_hook = tf.train.SummarySaverHook(
      save_steps=1000, summary_writer=summary_writer, summary_op=ops)
  step_counter_hook = tf.train.StepCounterHook(summary_writer=summary_writer)
  hooks = [stop_hook, step_counter_hook, summary_hook]

  sing_list = []
  sing = 0
  v_list = []
  p_list = []
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
      # config = config,
      log_step_count_steps=log_count,
      max_wait_secs=3600) as mon_sess:

    while not mon_sess.should_stop():
      # unused_var, loss_var, step_var, unused_var4, out_var, vert_var, smooth_var, points_var, overlap_var = mon_sess.run(
      #   [batch, train_loss, global_step, train_op, out_points, vert, smooth, points, overlap])
      unused_var, loss_var, step_var, unused_var4, vert_var, smooth_var, overlap_var = mon_sess.run(
        [batch, train_loss, global_step, train_op, vert, smooth, overlap])
      # distance_var = np.min(np.abs(distance_var), axis = 1)
      if step_var % 100 == 0:
        print("")
        print("Step: ", step_var, "\t\tLoss: ", loss_var)
        # print("Time: ", dt_var)
        # print("Smoothness: ", smooth_var[0, :])
        # print("Growth Ratio: ", overlap_var[0, :, 0])
        # print("Distances: ", np.array([np.min(distance_var), np.max(distance_var)]))
        # print("Undef: ", undef_dist_var[0, 0, np.argmin(distance_var[0, :, 0]), 9], undef_dist_var[0, 0, np.argmin(distance_var[0, :, 0]), 10], undef_dist_var[0, 0, np.argmin(distance_var[0, :, 0]), 11])
        # print("Undef: ", undef_dist_var[0, 1, np.argmin(distance_var[0, :, 0]), 9], undef_dist_var[0, 1, np.argmin(distance_var[0, :, 0]), 10], undef_dist_var[0, 1, np.argmin(distance_var[0, :, 0]), 11])
        # print("Retraction Quantile: ", np.quantile(retraction_var[0, :, 0], 0.25), np.quantile(retraction_var[0, :, 0], 0.5), np.quantile(retraction_var[0, :, 0], 0.75))
        # print("Retraction Mean: ", np.mean(retraction_var[0, :, 0]))
        # print("Distance Divergence: ", np.sum((distance_var > 0.1)*1))
        # print("Retraction Points: ", np.array([points_var[0, np.argmin(retraction_var[0, :, 0]), :], points_var[0, np.argmax(retraction_var[0, :, 0]), :]]))
        # print("Loop Iter: ", iter_var)
        # print("Distance Loop Iter: ", iter_dist_var)
        # print("Undef: ", undef_var)
        # print("Retraction Undef: ", undef_ret_var)
        # print("Retraction Undef: ", undef_ret_var[0, :, np.argmin(retraction_var[0, :, 0]), :], undef_ret_var[0, :, np.argmax(retraction_var[0, :, 0]), :])
        # print("Retraction Undef: ", undef_ret_var[0, :, np.argmin(retraction_var[0, :, 0]), 0, :], undef_ret_var[0, :, np.argmax(retraction_var[0, :, 0]), 0, :])
        print("")
        sing_list.append(sing)
        sing = 0
      # if undef_var[0, 0, -1, 0] != 0:
      #   print("")
      #   print("----------Singularity----------")
      #   print("Step: ", step_var, "\t\tLoss: ", loss_var)
      #   # print("Smoothness: ", smooth_var[0, :])
      #   # print("Growth Ratio: ", overlap_var[0, :, 0])
      #   # print("Loop Iter: ", iter_var)
      #   # print("Undef: ", undef_var)
      #   print("")
      #   sing += 1
      if step_var % 1000 == 0:
        v_list.append(vert_var[0, ...].reshape(-1, 3))
        p_list.append(smooth_var[0, :])

      if step_var >= FLAGS.max_steps - 1:
        sing_list = np.array(sing_list)
        print("")
        print("Singularity result: ", sing_list)
        print("")
        
        # out_var = out_var[0, :, :]
        smooth_var = smooth_var[0, :]
        vert_var = vert_var[0, ...].reshape(-1, 3)
        # points_var = points_var[0, ...]
        overlap_var = overlap_var[0, :, 0]
        with tf.io.gfile.GFile(path.join(FLAGS.train_dir, "stats.csv"), "w") as fout:
          fout.write("p, , \n")
          for i in range(smooth_var.shape[0]):
            fout.write("{0},{1},{2}\n".format(smooth_var[i], 0.0, 0.0))
          fout.write("vertex\n")
          for i in range(vert_var.shape[0]):
            fout.write("{0},{1},{2}\n".format(vert_var[i, 0], vert_var[i, 1], vert_var[i, 2]))
          # fout.write("dataset\n")
          # for i in range(points_var.shape[0]):
          #   fout.write("{0},{1},{2}\n".format(points_var[i, 0], points_var[i, 1], points_var[i, 2]))
          fout.write("overlap\n")
          for i in range(overlap_var.shape[0]):
            fout.write("{}\n".format(overlap_var[i]))
          # fout.write("x,y,z\n")
          # for i in range(out_var.shape[0]):
          #   fout.write("{0},{1},{2}\n".format(out_var[i, 0], out_var[i, 1], out_var[i, 2]))

        v_list = np.array(v_list)
        p_list = np.array(p_list)
        with tf.io.gfile.GFile(path.join(FLAGS.train_dir, "v_p.csv"), "w") as fout:
          fout.write("size,{0},{1}\n".format(0.0, 0.0))
          fout.write("{0},{1},{2}\n".format(v_list.shape[0], 0.0, 0.0))
          fout.write("p,{0},{1}\n".format(0.0, 0.0))
          for i in range(p_list.shape[0]):
            for j in range(p_list.shape[1]):
              fout.write("{0},{1},{2}\n".format(p_list[i, j], 0.0, 0.0))
          fout.write("vertex\n")
          for i in range(v_list.shape[0]):
            for j in range(v_list.shape[1]):
              fout.write("{0},{1},{2}\n".format(v_list[i, j, 0], v_list[i, j, 1], v_list[i, j, 2]))

          # fout.write("Minimum 0\n")
          # fout.write("{0},{1},{2}\n".format(undef_dist_var[0, 0, np.argmin(distance_var[0, :, 0]), 0, 21], undef_dist_var[0, 0, np.argmin(distance_var[0, :, 0]), 1, 21], undef_dist_var[0, 0, np.argmin(distance_var[0, :, 0]), 2, 21]))
          # fout.write("{0},{1},{2}\n".format(undef_dist_var[0, 0, np.argmin(distance_var[0, :, 0]), 0, 22], undef_dist_var[0, 0, np.argmin(distance_var[0, :, 0]), 1, 22], undef_dist_var[0, 0, np.argmin(distance_var[0, :, 0]), 2, 22]))
          # fout.write("{0},{1},{2}\n".format(undef_dist_var[0, 0, np.argmin(distance_var[0, :, 0]), 0 ,23], undef_dist_var[0, 0, np.argmin(distance_var[0, :, 0]), 1, 23], undef_dist_var[0, 0, np.argmin(distance_var[0, :, 0]), 2 ,23]))
          # fout.write("Minimum 1\n")
          # fout.write("{0},{1},{2}\n".format(undef_dist_var[0, 1, np.argmin(distance_var[0, :, 0]), 0, 21], undef_dist_var[0, 1, np.argmin(distance_var[0, :, 0]), 1, 21], undef_dist_var[0, 1, np.argmin(distance_var[0, :, 0]), 2, 21]))
          # fout.write("{0},{1},{2}\n".format(undef_dist_var[0, 1, np.argmin(distance_var[0, :, 0]), 0, 22], undef_dist_var[0, 1, np.argmin(distance_var[0, :, 0]), 1, 22], undef_dist_var[0, 1, np.argmin(distance_var[0, :, 0]), 2, 22]))
          # fout.write("{0},{1},{2}\n".format(undef_dist_var[0, 1, np.argmin(distance_var[0, :, 0]), 0, 23], undef_dist_var[0, 1, np.argmin(distance_var[0, :, 0]), 1, 23], undef_dist_var[0, 1, np.argmin(distance_var[0, :, 0]), 2, 23]))
          # fout.write("Maximum 0\n")
          # fout.write("{0},{1},{2}\n".format(undef_dist_var[0, 0, np.argmax(distance_var[0, :, 0]), 0, 21], undef_dist_var[0, 0, np.argmax(distance_var[0, :, 0]), 1, 21], undef_dist_var[0, 0, np.argmax(distance_var[0, :, 0]), 2, 21]))
          # fout.write("{0},{1},{2}\n".format(undef_dist_var[0, 0, np.argmax(distance_var[0, :, 0]), 0, 22], undef_dist_var[0, 0, np.argmax(distance_var[0, :, 0]), 1, 22], undef_dist_var[0, 0, np.argmax(distance_var[0, :, 0]), 2, 22]))
          # fout.write("{0},{1},{2}\n".format(undef_dist_var[0, 0, np.argmax(distance_var[0, :, 0]), 0, 23], undef_dist_var[0, 0, np.argmax(distance_var[0, :, 0]), 1, 23], undef_dist_var[0, 0, np.argmax(distance_var[0, :, 0]), 2, 23]))
          # fout.write("Maximum 1\n")
          # fout.write("{0},{1},{2}\n".format(undef_dist_var[0, 1, np.argmax(distance_var[0, :, 0]), 0, 21], undef_dist_var[0, 1, np.argmax(distance_var[0, :, 0]), 1, 21], undef_dist_var[0, 1, np.argmax(distance_var[0, :, 0]), 2, 21]))
          # fout.write("{0},{1},{2}\n".format(undef_dist_var[0, 1, np.argmax(distance_var[0, :, 0]), 0, 22], undef_dist_var[0, 1, np.argmax(distance_var[0, :, 0]), 1, 22], undef_dist_var[0, 1, np.argmax(distance_var[0, :, 0]), 2, 22]))
          # fout.write("{0},{1},{2}\n".format(undef_dist_var[0, 1, np.argmax(distance_var[0, :, 0]), 0, 23], undef_dist_var[0, 1, np.argmax(distance_var[0, :, 0]), 1, 23], undef_dist_var[0, 1, np.argmax(distance_var[0, :, 0]), 2, 23]))
          
          # for j in range(21):
          #   fout.write("Iter {}\n".format(j))
          #   fout.write("{0},{1},{2}\n".format(undef_dist_var[0, 0, np.argmax(distance_var[0, :, 0]), 0, j], undef_dist_var[0, 0, np.argmax(distance_var[0, :, 0]), 1, j], undef_dist_var[0, 0, np.argmax(distance_var[0, :, 0]), 2, j]))
          #   fout.write("{0},{1},{2}\n".format(undef_dist_var[0, 1, np.argmax(distance_var[0, :, 0]), 0, j], undef_dist_var[0, 1, np.argmax(distance_var[0, :, 0]), 1, j], undef_dist_var[0, 1, np.argmax(distance_var[0, :, 0]), 2, j]))



          # for j in range(2):
          #   fout.write("Maximum Data {}\n".format(j))
          #   fout.write("initial direction convex {0} iter {1}\n".format(j, i))
          #   fout.write("{0},{1},{2},{3}\n".format(undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 0, 17, 0], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 1, 17, 0], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 2, 17, 0], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 3, 17, 0]))
          #   for i in range(10):
          #     fout.write("Trust region convex {0} iter {1}\n".format(j, i))
          #     fout.write("{}\n".format(undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 0, 0, i]))
          #     fout.write("dD convex {0} iter {1}\n".format(j, i))
          #     fout.write("{0},{1},{2},{3}\n".format(undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 0, 1, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 1, 1, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 2, 1, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 3, 1, i]))
          #     fout.write("Jacobian convex {0} iter {1}\n".format(j, i))
          #     for k in range(4):
          #       fout.write("{0},{1},{2},{3}\n".format(undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), k, 2, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), k, 3, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), k, 4, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), k, 5, i]))
          #     fout.write("Inverse jacobian convex {0} iter {1}\n".format(j, i))
          #     for k in range(4):
          #       fout.write("{0},{1},{2},{3}\n".format(undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), k, 6, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), k, 7, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), k, 8, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), k, 9, i]))
          #     fout.write("F convex {0} iter {1}\n".format(j, i))
          #     fout.write("{0},{1},{2},{3}\n".format(undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 0, 10, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 1, 10, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 2, 10, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 3, 10, i]))
          #     fout.write("dN convex {0} iter {1}\n".format(j, i))
          #     fout.write("{0},{1},{2},{3}\n".format(undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 0, 11, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 1, 11, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 2, 11, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 3, 11, i]))
          #     fout.write("dC convex {0} iter {1}\n".format(j, i))
          #     fout.write("{0},{1},{2},{3}\n".format(undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 0, 12, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 1, 12, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 2, 12, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 3, 12, i]))
          #     fout.write("var convex {0} iter {1}\n".format(j, i))
          #     fout.write("{0},{1},{2},{3}\n".format(undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 0, 13, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 1, 13, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 2, 13, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 3, 13, i]))
          #     fout.write("residual convex {0} iter {1}\n".format(j, i))
          #     fout.write("{}\n".format(undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 0, 14, i]))
          #     fout.write("actual reduction convex {0} iter {1}\n".format(j, i))
          #     fout.write("{}\n".format(undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 0, 15, i]))
          #     fout.write("predict reduction convex {0} iter {1}\n".format(j, i))
          #     fout.write("{}\n".format(undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 0, 16, i]))
          
          # for j in range(2):
          #   fout.write("Maximum Data {}\n".format(j))
          #   for i in range(20):
          #     fout.write("W_ie convex {0} iter {1}\n".format(j, i))
          #     for k in range(3):
          #       fout.write("{0},{1},{2}\n".format(undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), k, 0, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), k, 1, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), k, 2, i]))
          #     fout.write("W_inv_ie convex {0} iter {1}\n".format(j, i))
          #     for k in range(3):
          #       fout.write("{0},{1},{2}\n".format(undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), k, 3, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), k, 4, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), k, 5, i]))
          #     fout.write("u_ie convex {0} iter {1}\n".format(j, i))
          #     fout.write("{0},{1},{2}\n".format(undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 0, 6, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 1, 6, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 2, 6, i]))
          #     fout.write("s convex {0} iter {1}\n".format(j, i))
          #     fout.write("{0},{1},{2}\n".format(undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 0, 7, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 1, 7, i], undef_ret_2_var[0, j, np.argmax(retraction_var[0, :, 0]), 2, 7, i]))

              


if __name__ == "__main__":
  tf.app.run(main)

