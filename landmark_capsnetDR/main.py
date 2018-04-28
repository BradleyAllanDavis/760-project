import tensorflow as tf
import numpy as np
import os
import csv
import time
from skimage.transform import resize
import matplotlib.pyplot as plt

from model import CapsNet
from config import args


#############  for different datasets, only need to change zz_config_face
#############                for all three .py files
###############          04/10/18   worked

# length = 2965     # for face96 dataset

def train(args, from_epoch=0):

    if not os.path.exists(os.path.dirname(args.summary_path)):
        os.mkdir(os.path.dirname(args.summary_path))
    acc_file = open(args.summary_path, 'a')

    tf.set_random_seed(100)
    imgs = tf.placeholder(tf.float32, [args.batch_size, args.height,
                                       args.width, args.num_channel])
    labels_raw = tf.placeholder(tf.float32, [args.batch_size])
    labels = tf.one_hot(tf.cast(labels_raw, tf.int32), args.num_class)

    model = CapsNet(imgs, labels, 'capsnet')
    model.build()
    model.calc_loss()
    pred = tf.argmax(tf.reduce_sum(tf.square(model.caps2),axis=[2,3]),axis=1)
    pred = tf.cast(pred, tf.float32)
    correct_pred = tf.cast(tf.equal(pred, labels_raw), tf.float32)
    accuracy = tf.reduce_mean(correct_pred)
    loss = model.total_loss

    optimizer = tf.train.AdamOptimizer(0.0001)
    train_op = optimizer.minimize(loss)

#########  will run out of memory
##    val_imgs = tf.placeholder(tf.float32, [10000, args.height, args.width, args.num_channel])
##    val_labels_raw = tf.placeholder(tf.float32, [10000])
##    val_labels = tf.one_hot(tf.cast(val_labels_raw, tf.int32), 10)
##    val_model = CapsNet(val_imgs, val_labels, name="capsnet")
##    val_model.build()
##    val_model.calc_loss()
##    val_pred = tf.argmax(tf.reduce_sum(tf.square(val_model.caps2),axis=[2,3]),axis=1)
##    val_pred = tf.cast(val_pred, tf.float32)
##    correct_val_pred = tf.cast(tf.equal(val_pred, val_labels_raw), tf.float32)
##    val_accuracy = tf.reduce_mean(correct_val_pred)
##    val_loss = val_model.total_loss

####################       for mnist data
##    train_img, train_label = prepare_data(args.data_path, 'mnist_train.csv', 60000)
##    train_img = np.reshape(train_img, [-1, 28, 28, 1])
##    num_train = train_img.shape[0]
##    test_img, test_label = prepare_data(args.data_path, 'mnist_test.csv', 10000)
##    test_img = np.reshape(test_img, [-1, 28, 28, 1])

############           for face96 data
##    data = np.load("face_96.npy")
##    total_label = data[:,0]
##    total_img = data[:,1:]/255
##    #total_img, total_label = prepare_data('face_96.csv', length)
##    test_img = total_img[[i for i in range(0, length, 5)],:]
##    test_label = total_label[[i for i in range(0, length, 5)]]
##    test_img = np.reshape(test_img, [-1, 196, 196, 3])
##    num_test = test_img.shape[0]
##    train_idx = list(set(range(length)) - set(range(0, length, 5)))
##    train_img = total_img[train_idx, :]
##    train_label = total_label[train_idx]
##    train_img = np.reshape(train_img, [-1, 196, 196, 3])
##    num_train = train_img.shape[0]
##
##    train_img_resize = np.zeros([length, 28, 28, 3])
##    for i in range(num_train):
##        train_img_resize[i,:,:,0] = resize(train_img[i,:,:,0],[28,28])
##        train_img_resize[i,:,:,1] = resize(train_img[i,:,:,1],[28,28])
##        train_img_resize[i,:,:,2] = resize(train_img[i,:,:,2],[28,28])
##    test_img_resize = np.zeros([length, 28, 28, 3])
##    for i in range(num_test):
##        test_img_resize[i,:,:,0] = resize(test_img[i,:,:,0],[28,28])
##        test_img_resize[i,:,:,1] = resize(test_img[i,:,:,1],[28,28])
##        test_img_resize[i,:,:,2] = resize(test_img[i,:,:,2],[28,28])
##    train_img = train_img_resize
##    test_img = test_img_resize

################   for landmark data, don't need this one
#######            idea is using getimagebatch and keep track of
##########         label using index of the labels.npy
##    #train_labels = np.load("labels_train_8249.npy")
##    num_labels= 50
##    train_size = 0.9
##    test_size = 0.1
##    data_meta = np.load(args.metadata_path)
##    # print(data_meta.shape)  (1225029,3)
##    (data_urls_train, labels_train, imgid_train, data_urls_test,
##     labels_test, imgid_test) = Utils_Data.FormatDataset(data_meta,
##              num_labels=num_labels, train_size=train_size, test_size=test_size)
##    num_train = data_urls_train.size
##    num_test = data_urls_test.size
##    #print(set(labels_train),set(labels_test))   ## 50 classes
##    labels_train_set = list(set(labels_train))
##    labels_train_dict = {labels_train_set[i]:i for i in range(len(labels_train_set))}
##    labels_train = [labels_train_dict[labels_train[i]] for i in range(len(labels_train))]
##    labels_train = np.array(labels_train)
##    #print(labels_train[:100])
##    np.random.seed(100)
##    np.random.shuffle(data_urls_train)
##    np.random.seed(100)
##    np.random.shuffle(labels_train)
##    np.random.seed(100)
##    np.random.shuffle(imgid_train)
##    in_memory_url = data_urls_train[:30000]
##    in_memory_labels = labels_train[:30000]
##    in_memory_imgid = imgid_train[:30000]
##    curr_index = 0
##    img_bch, curr_index, label_index = Utils_Data.GetImageBatch(urls=in_memory_url,
##                    start_index=curr_index,imgids=in_memory_imgid, batch_size=30000, \
##                    path='../images', n_rows=28, n_cols=28)
##    print(img_bch.shape, label_index[:100])
##    raise
##

######   load train and test data, trainsize ~30000, testsize ~34000

    test_img = np.load(os.path.join(args.data_path, "train_img_9.npy"))
    test_label = np.load(os.path.join(args.data_path, "train_label_9.npy"))
    num_test = test_img.shape[0]

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        if from_epoch == 0:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, args.CheckPointPath+"-{}".format(from_epoch))
        for epoch in range(from_epoch, from_epoch+args.epochs):
            if epoch % 10 == 0:
                print("-----------------loading data--------------------")
                train_img = np.load(os.path.join(args.data_path,
                    "train_img_{}.npy".format(epoch//10%9)))
                print("check: ", train_img.max(), train_img.shape)
                train_label = np.load(os.path.join(args.data_path,
                    "train_label_{}.npy".format(epoch//10%9)))
                print("check: ", train_label.max(), train_label.shape)
                num_train = train_img.shape[0]
            loops = num_train//args.batch_size
            curr_index = 0
            for i in range(loops):
                tic = time.time()
                #### get input mini-batch
                img_bch = train_img[(i*args.batch_size):((i+1)*args.batch_size),:,:,:]
                label_bch = train_label[(i*args.batch_size):((i+1)*args.batch_size)]

                #######   this method need IO from disk, slow
                #img_bch, curr_index, label_index = Utils_Data.GetImageBatch(urls=data_urls_train,
                #    start_index=curr_index,imgids=imgid_train, batch_size=args.batch_size, \
                #    path='../images', n_rows=28, n_cols=28)
                #print(curr_index, label_index)
                #img_bch = np.array(img_bch / 255 * 2 - 2, np.float32)
                #label_bch = np.array(labels_train[label_index], np.float32)

                #### train the model, take down the current training accuracy
                _, cur_accuracy, cur_loss = sess.run([train_op, accuracy, loss],
                                                     feed_dict={imgs: img_bch,
                                                     labels_raw: label_bch})
                print("Epoch: {} BATCH: {} Time: {:.4f} Loss:{:.4f} Accu: {:.4f}".format(
                      epoch, i, time.time()-tic, cur_loss, cur_accuracy))


            if epoch % 30 == 0:
                saver.save(sess, args.CheckPointPath, global_step=epoch)

            #### get the validate accuracy
            acc = []
            for j in range(num_test//args.batch_size//10):
                val_img_bch = test_img[(j*args.batch_size):((j+1)*args.batch_size),:,:,:]
                val_label_bch = test_label[(j*args.batch_size):((j+1)*args.batch_size)]

                val_acc = sess.run(accuracy,
                                   feed_dict={imgs: val_img_bch,
                                              labels_raw: val_label_bch})
                acc.append(val_acc)

            print("Epoch: {} TestAccu: {:.4f}".format(
                  epoch, np.mean(acc)))
            try:
                acc_file.write("Epoch: {} TestAccu: {:.4f}\n".format(
                           epoch, np.mean(acc)))
                acc_file.flush()
            except:
                continue
    acc_file.close()

#################   face96 prepare data
##def prepare_data(name, length):
##    ''' read csv data, length is 2965'''
##    with open(name) as file:
##        data_iter = csv.reader(file)
##        #next(data_iter)
##        data = [next(data_iter) for i in range(length)]
##    data = np.asarray(data, dtype=np.float32)
##    #labels = data[:,0]
##    #imgs = data[:,1:]/255
##    print('imgs loaded')
##    np.save("face_96.npy", data)
##    #return imgs, labels

################   mnist prepare data
##def prepare_data(path, name, length):
##    ''' read csv data'''
##    with open(os.path.join(path, name)) as file:
##        data_iter = csv.reader(file)
##        #next(data_iter)
##        data = [next(data_iter) for i in range(length)]
##    data = np.asarray(data, dtype=np.float32)
##    labels = data[:,0]
##    imgs = data[:,1:]/255
##    return imgs, labels
##a,b = prepare_data('face_96.npy', 2965)


if __name__ == "__main__":
    #data = np.load("face_96.npy")
    train(args, 0)
