##### Native libraries
import sys, os, csv
from PIL import Image
from io import BytesIO
from urllib import request
##### Python Libraries
import numpy as np
from IPython.core import debugger
breakpoint = debugger.set_trace
##### Local libraries

##### Format dataset
## Overview: Finds the top num_labels classes with the most number of samples. Then it splits into
## 		train and test sets according to train_size and test_size variables.
## Inputs: 
##		* dataset: n x m numpy matrix. n samples, with m features. The last column should be the labels
##		* num_labels: Out of the total number of labels L how many do you want to use in the dataset
##		* train_size: Percent of the entries to put in the train set 
##		* test_size: Percent of the entries to put in the test set 
def FormatDataset(dataset, num_labels=10, train_size=0.9, test_size=0.1):
	##### Get labels and sort entries to make formatting faster
	## Assume the labels are in the last column
	labels = dataset[:,-1].astype(int)
	sorting_indeces = np.argsort(labels)
	sorted_labels = labels[sorting_indeces]
	sorted_dataset = dataset[sorting_indeces, :]
	##### Find all unique labels in the sorted array. This lets us find the labels with the most
	## samples.
	(unique_labels, label_counts) = np.unique(labels, return_counts=True)
	## Sort the label counts in descending order and get the indeces for this sorting
	max_sample_label_indeces = np.argsort(label_counts)[::-1]
	## Sort label counts and the unique labels accordingly. Only use top n labels.
	label_counts = label_counts[max_sample_label_indeces[0:num_labels]]
	unique_labels = unique_labels[max_sample_label_indeces[0:num_labels]]
	##### Reduce dataset and labels so that they only include the labels with the most counts
	## Keep track of the number of train and test samples. 
	n_samples_train = 0
	n_samples_test = 0
	## Get the total number of samples
	n_samples = np.sum(label_counts)
	n_label_dataset = np.empty(shape=(n_samples,), dtype='|S233')
	n_label_ids = np.empty(shape=(n_samples,), dtype='|S233')
	n_labels = np.zeros((n_samples,), dtype=int)
	(start_index1,end_index1,start_index2,end_index2) = (0,0,0,0) 
	for i in range(num_labels):
		curr_label = unique_labels[i]
		## Get start and end indeces for the sorted labels array
		start_index1 = np.searchsorted(sorted_labels, curr_label, side='left')
		end_index1 = np.searchsorted(sorted_labels, curr_label, side='right')
		## Get start and end indeces for the elements we are adding
		start_index2 = end_index2
		end_index2 = start_index2 + label_counts[i]
		## Add elements to the n_label lists
		n_label_ids[start_index2:end_index2] = sorted_dataset[start_index1:end_index1,0]
		n_label_dataset[start_index2:end_index2] = sorted_dataset[start_index1:end_index1,1]
		n_labels[start_index2:end_index2] = sorted_labels[start_index1:end_index1]
		##### Keep track of the size of the training and testing set. This makes splitting the 
		## dataset much faster and easier.
		if(label_counts[i] == 1):
			n_samples_train = n_samples_train + 1
		else:
			n_samples_train = n_samples_train + np.int(np.floor(train_size*label_counts[i]))
			n_samples_test = n_samples_test + (label_counts[i] - np.int(np.floor(train_size*label_counts[i])))
	breakpoint()
	##### Find all unique labels in the sorted array. Allows to pre
	# (unique_labels, label_counts) = np.unique(sorted_labels, return_counts=True)
	##### Indexing variables 
	# curr_index = 0
	# start_index = 0
	# end_index = 0
	# for i in range(num_labels):
	# 	curr_label = unique_labels[i]
	# 	start_index = end_index
	# 	end_index = start_index + label_counts[i]
	# 	n_label_ids[start_index:end_index] = sorted_dataset[start_index:end_index,0]
	# 	n_label_dataset[start_index:end_index] = sorted_dataset[start_index:end_index,1]
	# 	n_labels[start_index:end_index] = sorted_labels[start_index:end_index]
	# 	##### Keep track of the size of the training and testing set. This makes splitting the 
	# 	## dataset much faster and easier.
	# 	if(label_counts[i] == 1):
	# 		n_samples_train = n_samples_train + 1
	# 	else:
	# 		n_samples_train = n_samples_train + np.int(np.floor(train_size*label_counts[i]))
	# 		n_samples_test = n_samples_test + (label_counts[i] - np.int(np.floor(train_size*label_counts[i])))
	print("The first {} labels have {} samples".format(num_labels,n_labels.size))
	##### Find all unique labels
	(unique_labels, label_counts) = np.unique(n_labels, return_counts=True)
	##### Pre-allocate the arrays for the train and test datasets and their labels
	n_label_train_ids = np.empty(shape=(n_samples_train,),dtype='|S233')
	n_label_train_dataset = np.empty(shape=(n_samples_train,),dtype='|S233')
	n_labels_train = np.empty(shape=(n_samples_train,),dtype=int)
	n_label_test_ids = np.empty(shape=(n_samples_test,),dtype='|S233')
	n_label_test_dataset = np.empty(shape=(n_samples_test,),dtype='|S233')
	n_labels_test = np.empty(shape=(n_samples_test,),dtype=int)
	print('n_train samples: {}, n_test samples: {}'.format(n_samples_train,n_samples_test))
	##### 
	curr_index = 0
	curr_train_index = 0
	curr_test_index = 0
	##### For each unique label split evenly for train and test
	for i in range(unique_labels.size):
		#####
		n_samples = label_counts[i]
		n_train = np.int(np.floor(train_size*n_samples))
		##### Do the indexing arithmetic. If there is only a single sample with that index we 
		## place it in the train set. If there is more of than one sample at least 1 sample will go
		## to the test set. If there is many samples for that class we will try to split proportional
		## to train_size and test_size.
		start_index_train = curr_index
		if(n_samples==1): end_index_train = start_index_train + 1
		else: end_index_train = start_index_train + n_train
		start_index_test = end_index_train 
		if(n_samples == 1): end_index_test = start_index_test 
		else: end_index_test = start_index_test + (n_samples - n_train) 
		#####
		## Data urls
		n_label_train_dataset[curr_train_index:curr_train_index+(end_index_train-start_index_train)] = n_label_dataset[start_index_train:end_index_train]
		n_label_test_dataset[curr_test_index:curr_test_index+(end_index_test-start_index_test)] = n_label_dataset[start_index_test:end_index_test]
		## ids
		n_label_train_ids[curr_train_index:curr_train_index+(end_index_train-start_index_train)] = n_label_ids[start_index_train:end_index_train]
		n_label_test_ids[curr_test_index:curr_test_index+(end_index_test-start_index_test)] = n_label_ids[start_index_test:end_index_test]
		## Labels
		n_labels_train[curr_train_index:curr_train_index+(end_index_train-start_index_train)] = n_labels[start_index_train:end_index_train]
		n_labels_test[curr_test_index:curr_test_index+(end_index_test-start_index_test)] = n_labels[start_index_test:end_index_test]
		#####
		curr_train_index = curr_train_index+(end_index_train-start_index_train)
		curr_test_index = curr_test_index+(end_index_test-start_index_test)
		curr_index = end_index_test
	print("Train set size: {}, Test set size: {}".format(n_labels_train.size,n_labels_test.size))
	##### Remove some extra characters from each string
	for i in range(n_label_train_dataset.size): 
		n_label_train_dataset[i] = n_label_train_dataset[i][1:-1] 
		n_label_train_ids[i] = n_label_train_ids[i][1:-1] 
	for i in range(n_label_test_dataset.size): 
		n_label_test_dataset[i] = n_label_test_dataset[i][1:-1] 
		n_label_test_ids[i] = n_label_test_ids[i][1:-1] 

	return (n_label_train_dataset,n_labels_train,n_label_train_ids,n_label_test_dataset,n_labels_test,n_label_test_ids)

##### Downloads image in the url
## If download fails it returns None
def DownloadImage(url, imgid=0):
	pil_image_rgb = None
	try:
		response = request.urlopen(url.decode('UTF-8'))
		image_data = response.read()
	except:
		print('Warning: Could not download image %s from %s' % (imgid, url.decode('UTF-8')))
		return pil_image_rgb
	try:
	# pil_image = Image.open(StringIO(image_data))
	# pil_image = Image.open(cStringIO.StringIO(image_data))
		pil_image = Image.open(BytesIO(image_data))
	except:
		print('Warning: Failed to parse image %s' % imgid)
		return pil_image_rgb
	try:
		pil_image_rgb = pil_image.convert('RGB')
	except:
		print('Warning: Failed to convert image %s to RGB' % imgid)
		return pil_image_rgb
	return pil_image_rgb

##### Resize image and cast to numpy array
## Even though it is only two lines we want to make sure that we do this step in the exact
## same way everywhere
def FormatImage(image,n_rows,n_cols):
	resized_image = image.resize((n_rows,n_cols), resample=Image.ANTIALIAS)
	I = np.asarray(resized_image)
	return (I, resized_image)

##### Downloads the image (if we have not downloaded it) resizes it and saves it
def DownloadResizeAndSave(url, out_dir, imgid=0, n_cols=256, n_rows=256):
	filename = os.path.join(out_dir, '{}'.format(imgid))
	filename_resized = os.path.join(out_dir, '{}_{}_{}'.format(imgid, n_rows, n_cols))
	I = np.array(())
	resized_image = None
	if os.path.exists(filename_resized+'.jpg'):
		# print('Image %s already exists. Skipping download.' % filename_resized)
		I = np.load(filename_resized+'.npy') 
	else:
		if os.path.exists(filename+'.jpg'):
			# print('Image %s already exists. Skipping download.' % filename)
			pil_image_rgb = Image.open(filename+'.jpg') 
		else:
			pil_image_rgb = DownloadImage(url, imgid=imgid)
			if(pil_image_rgb == None):
				return None
		(I, resized_image) = FormatImage(pil_image_rgb,n_rows=n_rows,n_cols=n_cols)
		resized_image.save(filename_resized+'.jpg', format='JPEG', quality=90)
		np.save(filename_resized, I)
	return I


def DownloadAndSaveImage(url, out_dir, imgid=0, resize=False, n_cols=0, n_rows=0):
	filename = os.path.join(out_dir, '%s.jpg' % imgid)
	I = np.array(())
	if os.path.exists(filename+'.jpg'):
		# print('Image %s already exists. Skipping download.' % filename)
		image_data = Image.open(filename) 
		I = np.asarray(image_data)
	else:
		pil_image_rgb = DownloadImage(url, imgid=imgid)
		if(pil_image_rgb == None):
			return I
		try:
			pil_image_rgb.save(filename, format='JPEG', quality=90)
		except:
			print('Warning: Failed to save image %s' % filename)
			return I
		I = np.asarray(pil_image_rgb)
	return I

def GetImage(url, imgid=0, path='./', n_rows=256, n_cols=256):
	##### Image to return
	I = np.array(())
	##### Get the resized image if it does not exist then we download it and resize it
	filename_resized = os.path.join(path, '{}_{}_{}'.format(imgid, n_rows, n_cols))
	if os.path.exists(filename_resized+'.npy'):
		# print('Image %s already exists. Skipping download.' % filename_resized)
		I = np.load(filename_resized+'.npy')
		return I
	else:
		##### Download and resize the image
		pil_image_rgb = DownloadImage(url, imgid=imgid)
		if(pil_image_rgb == None):
			return I
		(I, resized_image) = FormatImage(pil_image_rgb,n_rows=n_rows,n_cols=n_cols)
	return I

def GetImageBatch(urls, start_index, imgids=0, batch_size=4, path='./', n_rows=256, n_cols=256):
	##### Create arbitrary ids if none are given
	if(imgids==0): imgids = np.arange(0, urls.size)
	##### Allocate the image batch size
	I_batch = np.zeros((batch_size,n_rows,n_cols,3))
	##### counters to keep track of the number of images that have been loaded and
	## the current index in the urls that we are at
	num_loaded_images = 0
	curr_index = start_index
	##### We want to continue getting the batch until it is full or we ran out of urls
	while((num_loaded_images < batch_size) and (curr_index < urls.size)):
		I = GetImage(url=urls[curr_index], imgid=imgids[curr_index],\
									path=path, n_rows=n_rows, n_cols=n_cols)
		if(I.size != 0):
			I_batch[num_loaded_images,:,:,:] = I
			num_loaded_images = num_loaded_images+1
		curr_index = curr_index+1
	##### If the batch is not full resize it
	if(num_loaded_images != batch_size):
		I_batch = I_batch[0:num_loaded_images,:,:,:]
	##### Return the end_index
	end_index = curr_index
	return (I_batch, end_index)



