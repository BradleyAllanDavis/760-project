##### Native libraries

##### Python Libraries
import numpy as np
##### Local libraries
import Utils_Data
from Timer import Timer
from IPython.core import debugger
breakpoint = debugger.set_trace

##### NOTE: To download the full dataset (which will take about 30 hours on wifi maybe less on ethernet)
##### set the filename_urls to train.npy, set num_labels to 14951, set the 
##### for loop iterations to data_urls_train.size
##### Path to datasets
path_urls = '../../data/image_retrieval/image_recognition/'
filename = 'train'
csv_filename_urls = filename+'.csv' # Change this to train.npy to download the full dataset

urls_data = np.array(Utils_Data.parse_data(path_urls+csv_filename_urls),dtype='|S233')
np.save(path_urls+filename+'.npy',urls_data)


