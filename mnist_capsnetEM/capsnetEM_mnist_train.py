#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

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

def main(unused_argv):
  ###################################################################################
  # MNIST
  # (trX, trY) = load_mnist(is_training=True)
  # (teX, teY) = load_mnist(is_training=False)
  #### create lambda function that returns batch from mnist
  # train_inputs = get_create_inputs(dataset_name='mnist',is_train=True,epochs=10)
  """Get dataset hyperparameters."""
  dataset_name = "mnist"
  logger.info('Using dataset: {}'.format(dataset_name))
  """Set reproduciable random seed"""
  tf.set_random_seed(1234)

  coord_add = get_coord_add(dataset_name)
  dataset_size = get_dataset_size_train(dataset_name)
  num_classes = get_num_classes(dataset_name)
  create_inputs = get_create_inputs(dataset_name, is_train=True, epochs=cfg.epoch)

  with tf.Graph().as_default(), tf.device('/cpu:0'):
    """Get global_step."""
    global_step = tf.get_variable(
      'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    """Get batches per epoch."""
    num_batches_per_epoch = int(dataset_size / cfg.batch_size)

    """Use exponential decay leanring rate?"""
    lrn_rate = tf.maximum(tf.train.exponential_decay(
      1e-3, global_step, num_batches_per_epoch, 0.8), 1e-5)
    tf.summary.scalar('learning_rate', lrn_rate)
    opt = tf.train.AdamOptimizer()  # lrn_rate
    breakpoint()
    """Get batch from data queue."""
    batch_x, batch_labels = create_inputs() # get_create_inputs from capsnetEM_datautils.py
    # batch_y = tf.one_hot(batch_labels, depth=10, axis=1, dtype=tf.float32)
    # breakpoint()

    """Define the dataflow graph."""
    m_op = tf.placeholder(dtype=tf.float32, shape=())
    # with tf.device('/gpu:0'):
    with tf.device('/cpu:0'):
      with slim.arg_scope([slim.variable], device='/cpu:0'):
        # breakpoint()
        batch_squash = tf.divide(batch_x, 255.)
        batch_x = slim.batch_norm(batch_x, center=False, is_training=True, trainable=True)
        output, pose_out = build_arch(batch_x, coord_add, is_train=True,
                          num_classes=num_classes)
        # loss = cross_ent_loss(output, batch_labels)
        tf.logging.debug(pose_out.get_shape())
        loss, spread_loss_var, mse, _ = spread_loss(
          output, pose_out, batch_squash, batch_labels, m_op)
        acc = test_accuracy(output, batch_labels)
        tf.summary.scalar('spread_loss', spread_loss_var)
        tf.summary.scalar('reconstruction_loss', mse)
        tf.summary.scalar('all_loss', loss)
        tf.summary.scalar('train_acc', acc)

      """Compute gradient."""
      grad = opt.compute_gradients(loss)
      grad_check = [tf.check_numerics(g, message='Gradient NaN Found!') for g, _ in grad if g is not None] + [tf.check_numerics(loss, message='Loss NaN Found')]

    """Apply gradient."""
    with tf.control_dependencies(grad_check):
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grad, global_step=global_step)

    """Set Session settings."""
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    """Set Saver."""
    var_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]  
    saver = tf.train.Saver(var_list=var_to_save, max_to_keep=cfg.epoch)

    """Display parameters"""
    total_p = np.sum([np.prod(v.get_shape().as_list()) for v in var_to_save]).astype(np.int32)
    train_p = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]).astype(np.int32)
    logger.info('Total Parameters: {}'.format(total_p))
    logger.info('Trainable Parameters: {}'.format(train_p))

    # read snapshot
    # latest = os.path.join(cfg.logdir, 'model.ckpt-4680')
    # saver.restore(sess, latest)
    """Set summary op."""
    summary_op = tf.summary.merge_all()

    """Start coord & queue."""
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    """Set summary writer"""
    if not os.path.exists(cfg.logdir + '/caps/{}/train_log/'.format(dataset_name)):
      os.makedirs(cfg.logdir + '/caps/{}/train_log/'.format(dataset_name))
    summary_writer = tf.summary.FileWriter(
      cfg.logdir + '/caps/{}/train_log/'.format(dataset_name), graph=sess.graph)  # graph = sess.graph, huge!

    """Main loop."""
    m_min = 0.2
    m_max = 0.9
    m = m_min
    for step in range(cfg.epoch * num_batches_per_epoch + 1):
      tic = time.time()
      """"TF queue would pop batch until no file"""
      try:
        _, loss_value, summary_str = sess.run(
          [train_op, loss, summary_op], feed_dict={m_op: m})
        logger.info('%d iteration finished in ' % step + '%f second' %
              (time.time() - tic) + ' loss=%f' % loss_value)
      except KeyboardInterrupt:
        sess.close()
        sys.exit()
      except tf.errors.InvalidArgumentError:
        logger.warning('%d iteration contains NaN gradients. Discard.' % step)
        continue
      else:
        """Write to summary."""
        if step % 5 == 0:
          summary_writer.add_summary(summary_str, step)

        """Epoch wise linear annealling."""
        if (step % num_batches_per_epoch) == 0:
          if step > 0:
            m += (m_max - m_min) / (cfg.epoch * cfg.m_schedule)
            if m > m_max:
              m = m_max

          """Save model periodically"""
        if((step % (cfg.checkpoint_frequency*num_batches_per_epoch))==0):
          ckpt_path = os.path.join(
            cfg.logdir + '/caps/{}/'.format(dataset_name), 'model-{:.4f}.ckpt'.format(loss_value))
          saver.save(sess, ckpt_path, global_step=step)



if __name__ == "__main__":
  tf.app.run()













