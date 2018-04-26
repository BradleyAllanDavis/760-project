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
path_urls = '../../data/image_retrieval/image_recognition/'
save_path = path_urls + 'images/'
filename_urls = 'train.npy' # Change this to train.npy to download the full dataset
N_min = 5000

##### Load Full Dataset
dataset = np.load(path_urls+filename_urls)

##### Save a small version of the dataset
dataset_min = dataset[0:N_min,:]
np.save(path_urls+'train_{}.npy'.format(N_min),dataset_min)
np.savetxt(path_urls+'train_{}.csv'.format(N_min),dataset_min, fmt='%s,%s,%s',delimiter=',', newline='\n', header='"id","url","landmark_id"')

