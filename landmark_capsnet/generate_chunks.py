import numpy as np
import os
from skimage.transform import resize

import Utils_Data

num_labels = 50
train_size = 0.9
test_size = 0.1
data_meta = np.load("../train.npy")

chunk_size = 30000

def generate(): 
    '''generate chunks of npy datasets'''
    (data_urls_train, labels_train, imgid_train, data_urls_test,
     labels_test, imgid_test) = Utils_Data.FormatDataset(data_meta,
              num_labels=num_labels, train_size=train_size, test_size=test_size)
    num_train = data_urls_train.size
    num_test = data_urls_test.size
    
    labels_train_set = list(set(labels_train))
    labels_test_set = list(set(labels_test))
    print(labels_train, labels_test)
    ## standardize labels to [0,49]
    labels_dict = {labels_train_set[i]:i for i in range(len(labels_train_set))}
    labels_train = [labels_dict[label] for label in labels_train]
    labels_test = [labels_dict[label] for label in labels_test]
    labels_train = np.array(labels_train)
    labels_test = np.array(labels_test)

    np.random.seed(100)
    np.random.shuffle(labels_train)
    np.random.seed(100)
    np.random.shuffle(imgid_train)
    np.random.seed(100)
    np.random.shuffle(labels_test)
    np.random.seed(100)
    np.random.shuffle(imgid_test)

    def generate_chunk(img, label, i, name):
        in_memory_labels = label[i*chunk_size:(i+1)*chunk_size]
        in_memory_imgid = img[i*chunk_size:(i+1)*chunk_size]
        curr_index = 0
        img_resize = np.zeros(shape=[chunk_size,28,28,3])
        out_label = np.zeros(shape=[chunk_size])
        for j,img_id in enumerate(in_memory_imgid):
            try:
                print(curr_index)
                loc = os.path.join("../images/","{}_{}_{}".format(img_id,256,256))+".npy"
                img = np.load(loc)  ## (256,256,3)
                #out[curr_index] = img
                img_resize[curr_index] = resize(img,[28,28,3])
                out_label[curr_index] = in_memory_labels[j]
                curr_index += 1
            
            except FileNotFoundError:
                print("cannot load" + img_id.decode('utf-8'))
        print(curr_index)
        np.save("{}_img_{}.npy".format(name, i), img_resize[:curr_index])
        np.save("{}_label_{}.npy".format(name, i), out_label[:curr_index])

    ## generate train chunks
    for i in range(10):
        generate_chunk(imgid_train, labels_train, i, 'train')
    ## generate test chunk
    generate_chunk(imgid_test, labels_test, 0, 'test')
    
if __name__ == "__main__":
    generate()

