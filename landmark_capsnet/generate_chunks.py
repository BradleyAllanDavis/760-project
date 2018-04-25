import numpy as np
import os
from skimage.transform import resize

import Utils_Data

num_labels = 50
train_size = 0.9
test_size = 0.1
data_meta = np.load("../train.npy")
labels_dict = {}

def generate(): 
    '''generate chunks of npy datasets'''
    (data_urls_train, labels_train, imgid_train, data_urls_test,
     labels_test, imgid_test) = Utils_Data.FormatDataset(data_meta,
              num_labels=num_labels, train_size=train_size, test_size=test_size)
    num_train = data_urls_train.size
    num_test = data_urls_test.size
    
    labels_train_set = list(set(labels_train))
    labels_test = list(set(labels_test))
    print(labels_train_set, labels_test)

    global labels_dict
    labels_dict = {labels_train_set[i]:i for i in range(len(labels_train_set))}
    return 
    labels_train = [labels_train_dict[labels_train[i]] for i in range(len(labels_train))]
    labels_train = np.array(labels_train)
    np.random.seed(100)
    np.random.shuffle(data_urls_train)
    np.random.seed(100)
    np.random.shuffle(labels_train)
    np.random.seed(100)
    np.random.shuffle(imgid_train)
    in_memory_url = data_urls_train[:30000]
    in_memory_labels = labels_train[:30000]
    in_memory_imgid = imgid_train[:30000]
    curr_index = 0
    #out = np.zeros(shape=[30000,256,256,3])
    img_resize = np.zeros(shape=[30000,28,28,3])
    out_label = np.zeros(shape=[30000])
    for i,img_id in enumerate(in_memory_imgid):
        try:
            print(curr_index)
            loc = os.path.join("../images/","{}_{}_{}".format(img_id,256,256))+".npy"
            img = np.load(loc)  ## (256,256,3)
            #out[curr_index] = img
            img_resize[curr_index] = resize(img,[28,28,3])
            out_label[curr_index] = in_memory_labels[i]
            curr_index += 1
            
        except FileNotFoundError:
            print("cannot load" + img_id.decode('utf-8'))
    print(curr_index)
    #np.save("train_img_1.npy", out[:curr_index])
    np.save("train_img_resize_1.npy",img_resize[:curr_index])
    np.save("train_label_1.npy",out_label[:curr_index])

def generate_testset(): 
    '''generate testset'''
    (data_urls_train, labels_train, imgid_train, data_urls_test,
     labels_test, imgid_test) = Utils_Data.FormatDataset(data_meta,
              num_labels=num_labels, train_size=train_size, test_size=test_size)
    num_train = data_urls_train.size
    num_test = data_urls_test.size
    ## change labels to 0:49

    global labels_dict
    labels_test = [labels_dict[raw_label] for raw_label in labels_test]
    labels_test = np.array(labels_test)
    ## shuffle the data, otherwise first 10 labels are all 7.
    np.random.seed(100)
    np.random.shuffle(data_urls_test)
    np.random.seed(100)
    np.random.shuffle(labels_test)
    np.random.seed(100)
    np.random.shuffle(imgid_test)
    in_memory_url = data_urls_test
    in_memory_labels = labels_test
    in_memory_imgid = imgid_test
    curr_index = 0
    #out = np.zeros(shape=[30000,256,256,3])
    img_resize = np.zeros(shape=[num_test,28,28,3])
    out_label = np.zeros(shape=[num_test])
    for i,img_id in enumerate(in_memory_imgid):
        try:
            print(curr_index)
            loc = os.path.join("../test_images/","{}_{}_{}".format(img_id,256,256))+".npy"
            img = np.load(loc)  ## (256,256,3)
            #out[curr_index] = img
            img_resize[curr_index] = resize(img,[28,28,3])
            out_label[curr_index] = in_memory_labels[i]
            curr_index += 1
            
        except FileNotFoundError:
            print("cannot load" + img_id.decode('utf-8'))
    print(curr_index)
    #np.save("train_img_1.npy", out[:curr_index])
    np.save("test_img_resize.npy",img_resize[:curr_index])
    np.save("test_label.npy",out_label[:curr_index])

generate()
generate_testset()
