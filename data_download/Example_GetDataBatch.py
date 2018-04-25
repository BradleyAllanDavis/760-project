##### Native libraries

##### Python Libraries
import numpy as np
# from IPython.core import debugger
# breakpoint = debugger.set_trace
##### Local libraries
import Utils_Data
from Timer import Timer

##### NOTE: To download the full dataset (which will take about 30 hours on wifi maybe less on ethernet)
##### set the filename_urls to train.npy, set num_labels to 14951, set the 
##### for loop iterations to data_urls_train.size
##### Path to datasets
path_urls = '../data/image_retrieval/image_recognition/'
path_images = path_urls + 'images/'
filename_urls = 'train.npy' # Change this to train.npy to download the full dataset
##### Dataset format parameters
## Number of labels to use out of all the available ones 
## For train.npy (max = 14951)
## For train_100.npy (max = 79)
## For train_1000.npy (max = 692)
## For train_10000.npy (max = 3487)
num_labels=50
## Percent of entries to place in train set
train_size=0.9
## Percent of entries to place in test set
test_size=0.1
## Total number of images in batch
batch_size = 10
## Image dimensions
n_rows = 256
n_cols = 256
##### Load dataset
dataset = np.load(path_urls+filename_urls)
##### Split dataset in train and test containing the specified number of classes
## The following function returns all entries sorted for both train and test sets.
(data_urls_train, labels_train, imgid_train, data_urls_test, labels_test, imgid_test) = Utils_Data.FormatDataset(dataset, num_labels=num_labels, train_size=train_size, test_size=test_size) 
#####
## Curr index keeps track of where in the urls array we are at. Sometimes we will have to skip
## urls when loading a batch and this makes sure that we take into account that  
curr_index = 0
while(curr_index < data_urls_train.size):
	print("Start index: {}".format(curr_index))
	(I_batch, curr_index) = Utils_Data.GetImageBatch(data_urls_train, start_index=curr_index, \
										imgids=imgid_train, batch_size=batch_size, \
										path=path_images, n_rows=n_rows, n_cols=n_cols)
	print("End index: {}".format(curr_index))
	print("Curr Batch Size: {}".format(I_batch.shape[0]))

##### Save a small version of the dataset
# N_min = 10000
# dataset_min = dataset[0:N_min,:]
# np.save(path_urls+'train_min.npy',dataset_min)
# np.savetxt(path_urls+'train_min.csv',dataset_min, fmt='%s,%s,%s',delimiter=',', newline='\n', header='"id","url","landmark_id"')

