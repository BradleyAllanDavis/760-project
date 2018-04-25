##### Native libraries

##### Python Libraries
import numpy as np
from IPython.core import debugger
breakpoint = debugger.set_trace
##### Local libraries
import Utils_Data
from Timer import Timer

##### NOTE: To download the full dataset (which will take about 30 hours on wifi maybe less on ethernet)
##### set the filename_urls to train.npy, set num_labels to 14951, set the 
##### for loop iterations to data_urls_train.size
##### Path to datasets
path_urls = '../data/image_retrieval/image_recognition/'
save_path = path_urls + 'images/'
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
# breakpoint()
##### Load dataset
dataset = np.load(path_urls+filename_urls)
##### Split dataset in train and test containing the specified number of classes
## The following function returns all entries sorted for both train and test sets.
(data_urls_train, labels_train, imgid_train, data_urls_test, labels_test, imgid_test) = Utils_Data.FormatDataset(dataset, num_labels=num_labels, train_size=train_size, test_size=test_size) 
# breakpoint()
## Total number of images to download
# n_images = 20
n_images = data_urls_train.size
#####  
for i in range(0,n_images):
	with Timer('Download Image Time'):
		print("Image {} out of {}".format(i, n_images))
		# image = Utils_Data.DownloadAndSaveImage(url=data_urls_train[i],out_dir=save_path,imgid=imgid_train[i])
		image = Utils_Data.DownloadResizeAndSave(url=data_urls_train[i],out_dir=save_path,imgid=imgid_train[i])
