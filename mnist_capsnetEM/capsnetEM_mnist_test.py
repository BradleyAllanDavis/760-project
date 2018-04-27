#test.py

##### Pyhton Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import random
##### Library Imports
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time

import logging
import daiquiri

from IPython.core import debugger
breakpoint = debugger.set_trace
##### Local Imports
from capsnetEM_config import *
from capsnetEM_datautils import *
from capsnetEM_EMutils import *


daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


def main(args):
  dataset_name = "mnist"
  model_name = "caps"
  coord_add = get_coord_add(dataset_name)
  dataset_size_train = get_dataset_size_train(dataset_name)
  dataset_size_test = get_dataset_size_test(dataset_name)
  num_classes = get_num_classes(dataset_name)
  create_inputs = get_create_inputs(
    dataset_name, is_train=False, epochs=cfg.epoch)

  """Set reproduciable random seed"""
  tf.set_random_seed(1234)

  with tf.Graph().as_default():
    num_batches_per_epoch_train = int(dataset_size_train / cfg.batch_size)
    num_batches_test = int(dataset_size_test / cfg.batch_size * 0.1)

    batch_x, batch_labels = create_inputs()
    batch_x = slim.batch_norm(batch_x, center=False, is_training=False, trainable=False)
    if model_name == "caps":
      output, _ = build_arch(batch_x, coord_add,
                     is_train=False, num_classes=num_classes)
    elif model_name == "cnn_baseline":
      output = build_arch_baseline(batch_x,
                       is_train=False, num_classes=num_classes)
    else:
      raise "Please select model from 'caps' or 'cnn_baseline' as the secondary argument of eval.py!"
    batch_acc = test_accuracy(output, batch_labels)
    saver = tf.train.Saver()

    step = 0

    summaries = []
    summaries.append(tf.summary.scalar('accuracy', batch_acc))
    summary_op = tf.summary.merge(summaries)

    with tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)) as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      if not os.path.exists(cfg.test_logdir + '/{}/{}/'.format(model_name, dataset_name)):
        os.makedirs(cfg.test_logdir + '/{}/{}/'.format(model_name, dataset_name))
      summary_writer = tf.summary.FileWriter(
        cfg.test_logdir + '/{}/{}/'.format(model_name, dataset_name), graph=sess.graph)  # graph=sess.graph, huge!

      files = os.listdir(cfg.logdir + '/{}/{}/'.format(model_name, dataset_name))
      ckpt_frequency = cfg.checkpoint_frequency
      n_ckpts = int(np.round(cfg.epoch / ckpt_frequency))
      epochs = np.arange(0, n_ckpts)*num_batches_per_epoch_train*ckpt_frequency
      # for epoch in range(20, cfg.epoch):
      for epoch_id in epochs:
        # requires a regex to adapt the loss value in the file name here
        # ckpt_re = ".ckpt-%d" % (num_batches_per_epoch_train * epoch)
        ckpt_re = ".ckpt-%d" % (epoch_id)
        for __file in files:
          if __file.endswith(ckpt_re + ".index"):
            ckpt = os.path.join(cfg.logdir + '/{}/{}/'.format(model_name, dataset_name), __file[:-6])
        
        # ckpt = os.path.join(cfg.logdir, "model.ckpt-%d" % (num_batches_per_epoch_train * epoch))
        saver.restore(sess, ckpt)

        accuracy_sum = 0
        for i in range(num_batches_test):
          batch_acc_v, summary_str = sess.run([batch_acc, summary_op])
          print('%d batches are tested.' % step)
          summary_writer.add_summary(summary_str, step)

          accuracy_sum += batch_acc_v

          step += 1

        ave_acc = accuracy_sum / num_batches_test
        print('the average accuracy is %f' % ave_acc)

      coord.join(threads)


if __name__ == "__main__":
  tf.app.run()