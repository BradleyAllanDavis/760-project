import tensorflow as tf
import numpy as np
import os

from model import CapsNet
from config import args

def encode(data, model_ckpt, save_name):
    num_test = data.shape[0]
    imgs = tf.placeholder(tf.float32, [args.batch_size, args.height,
                                       args.width, args.num_channel])
    model = CapsNet(imgs, None, 'capsnet')
    model.build()

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        saver.restore(sess, model_ckpt)
        encode = np.zeros([num_test, args.num_class, 16])
        for i in range(num_test // args.batch_size):
            img_bch = data[i*args.batch_size:(i+1)*args.batch_size,:,:,:]
            encode_bch = sess.run(model.caps2, feed_dict={imgs:img_bch})
            assert encode_bch.shape == (args.batch_size, args.num_class, 16 ,1)
            encode[i*args.batch_size:(i+1)*args.batch_size,:,:] = \
                                             np.squeeze(encode_bch,[-1])
    np.save(save_name, encode)


for i in range(9,10):
    data = np.load("./landmark_data/train_img_{}.npy".format(i))
    encode(data, "./ckpt_0427/my_model.ckpt-210",
           "./landmark_data/encode_img_{}.npy".format(i))
