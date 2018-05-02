import os
import numpy as np
import tensorflow as tf

from IPython.core import debugger
breakpoint = debugger.set_trace

from capsnetEM_config import *

def create_inputs_mnist(is_train):
  tr_x, tr_y = load_mnist(is_train)
  data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
  x, y = tf.train.shuffle_batch(data_queue, num_threads=8, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)

  return (x, y)

def load_mnist(is_training):
  #### verify that we have downloaded the mnist dataset
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  test_data = mnist.test.images  # Returns np.array
  test_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  # fd = open(os.path.join(cfg.dataset, 'train-images-idx3-ubyte'))
  # loaded = np.fromfile(file=fd, dtype=np.uint8)
  trX = train_data.reshape((55000, 28, 28, 1)).astype(np.float32)

  # fd = open(os.path.join(cfg.dataset, 'train-labels-idx1-ubyte'))
  # loaded = np.fromfile(file=fd, dtype=np.uint8)
  trY = train_labels.reshape((55000)).astype(np.int32)

  # fd = open(os.path.join(cfg.dataset, 't10k-images-idx3-ubyte'))
  # loaded = np.fromfile(file=fd, dtype=np.uint8)
  teX = test_data.reshape((10000, 28, 28, 1)).astype(np.float32)

  # fd = open(os.path.join(cfg.dataset, 't10k-labels-idx1-ubyte'))
  # loaded = np.fromfile(file=fd, dtype=np.uint8)
  teY = test_labels.reshape((10000)).astype(np.int32)

  if is_training:
    return trX, trY
  else:
    return teX, teY
      
def get_create_inputs(dataset_name, is_train, epochs):
  options = {'mnist': lambda: create_inputs_mnist(is_train)}
  return options[dataset_name]
