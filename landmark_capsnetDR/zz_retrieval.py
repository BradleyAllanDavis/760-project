import numpy as np
from scipy.misc import imread, imsave
import os

query_data = np.load("./landmark_data/encode_img_0.npy")
query_label = np.load("./landmark_data/train_label_0.npy")
query_id = np.load("./landmark_data/train_imgid_0.npy")
test_data = np.load("./landmark_data/encode_img_9.npy")
print('test_data shape: ', test_data.shape)
test_label = np.load("./landmark_data/train_label_9.npy")
test_id = np.load("./landmark_data/train_imgid_9.npy")

def calc_distance(img, test_data):
    '''img is an np array 50*16, test_data is also np array N*50*16'''
    distance = np.zeros(test_data.shape[0])
    for i in range(test_data.shape[0]):
        distance[i] = np.linalg.norm((img - test_data[i]), 'fro')
    return distance

# for saving results
if not os.path.exists("./retrieval"):
    os.mkdir("./retrieval")
    
query = {}
query_with_id = {}
i = 0
while len(query) <= 49:
    if query_label[i] not in query:
        query[query_label[i]] = query_data[i]
        query_with_id[query_label[i]] = query_id[i]
    i += 1

accuracy_lst = []
accuracy_file = open("retrieval_accuracy.txt", 'a')
accuracy_file.write("top-5 accuracy\n")
for label, img in query.items():
    distance = calc_distance(img, test_data)
    rank = sorted(range(len(distance)), key=lambda x: distance[x])
    
    sorted_label = test_label[rank]
    # print("top 50 labels: ", label, sorted_label[:5])
    
    accuracy = np.sum(label == sorted_label[:5]) / 5
    accuracy_lst.append(accuracy)
    print("label {}, accuracy {}".format(label, accuracy))
    accuracy_file.write("label {}, accuracy {}\n".format(label, accuracy))
    
    #using test_id get top 5 images
    query_img = imread("../images/{}_256_256.jpg".format(query_with_id[label]))
    imsave("./retrieval/{}_q.jpg".format(int(label)), query_img)
    for i in range(5):
        img_id = test_id[rank[i]]
        retrieve_img = np.load("../images/{}_256_256.npy".format(img_id))
        imsave("./retrieval/{}_{}.jpg".format(int(label), i), retrieve_img)

print("mean accuracy: {}".format(np.mean(accuracy_lst)))
accuracy_file.write("mean accuracy: {}\n".format(np.mean(accuracy_lst)))
accuracy_file.flush()
accuracy_file.close()
