import os
import scipy
import numpy as np
import tensorflow as tf

from config import cfg

import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

# from IPython.core import debugger
# breakpoint = debugger.set_trace


def create_inputs_mnist(is_train):
    tr_x, tr_y = load_mnist(cfg.dataset, is_train)
    data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
    x, y = tf.train.shuffle_batch(data_queue, num_threads=8, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)

    return (x, y)

def create_inputs_landmark(is_train, chunk_id=0):
    tr_x, tr_y = load_landmark(cfg.dataset, is_train,chunk_id=chunk_id)
    data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
    x, y = tf.train.shuffle_batch(data_queue, num_threads=8, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)

    return (x, y)


def load_mnist(path, is_training):
    batch_size = 10000

    fd = open(os.path.join(cfg.dataset, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)
    trX = trX[0:batch_size,:,:,:]
    # print(trX.shape)

    fd = open(os.path.join(cfg.dataset, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int32)
    trY = trY[0:batch_size]

    fd = open(os.path.join(cfg.dataset, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)
    teX = teX[0:batch_size,:,:,:]

    fd = open(os.path.join(cfg.dataset, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.int32)
    teY = teY[0:batch_size]

    # normalization and convert to a tensor [60000, 28, 28, 1]
    trX = tf.convert_to_tensor(trX / 255., tf.float32)
    teX = tf.convert_to_tensor(teX / 255., tf.float32)

    # => [num_samples, 10]
    # trY = tf.one_hot(trY, depth=10, axis=1, dtype=tf.float32)
    # teY = tf.one_hot(teY, depth=10, axis=1, dtype=tf.float32)

    if is_training:
        return trX, trY
    else:
        return teX, teY


def load_landmark(path, is_training, chunk_id = 0):
    trX = np.load(os.path.join(cfg.dataset, 'train_img_{}.npy'.format(chunk_id))).astype(np.float)
    trY = np.load(os.path.join(cfg.dataset, 'train_label_{}.npy'.format(chunk_id))).astype(np.int32)
    nrX = trX.shape[0]
    teX = np.load(os.path.join(cfg.dataset, 'train_img_{}.npy'.format(chunk_id))).astype(np.float)
    teY = np.load(os.path.join(cfg.dataset, 'train_label_{}.npy'.format(chunk_id))).astype(np.int32)
    neX = teX.shape[0]

    # For grayscale
    # If this doesnt not work, check that data is normalized
    trX = tf.convert_to_tensor(trX[:,:,:,1].reshape((nrX, 28, 28, 1)) , tf.float32)
    teX = tf.convert_to_tensor(teX[:,:,:,1].reshape((neX, 28, 28, 1)) , tf.float32)
    # breakpoint()
    # np.max(trX)
    # print(trX.shape)

    # For RGB
    # trX = tf.convert_to_tensor(trX ,tf.float32)
    # teX = tf.convert_to_tensor(teX ,tf.float32)

    if is_training:
        return trX, trY
    else:
        return teX, teY

def get_landmark_dataset_size(path, is_training, chunk_ids):
    dataset_size = 0
    for chunk_id in chunk_ids:
        (X,Y) = load_landmark(path,is_training,chunk_id)
        dataset_size = dataset_size + Y.shape[0]
    return dataset_size

def get_chunk_size(path, is_training, chunk_id):
    (X,Y) = load_landmark(path,is_training,chunk_id)
    chunk_size = Y.shape[0]
    return chunk_size

