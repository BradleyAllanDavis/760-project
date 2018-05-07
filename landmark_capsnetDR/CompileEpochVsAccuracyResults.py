import numpy as np


fname = 'summary_mnist.txt'
out_fname = 'test.csv'
if(fname=='summary.txt'):
	out_fname = 'epoch_accuracy_results_landmark.csv'
elif(fname=='summary_mnist.txt'):
	out_fname = 'epoch_accuracy_results_mnist.csv'
		

with open(fname) as f:
	## Get full content of file
	content = f.readlines()
	# print(content)
	## Number of lines is number of epochs
	n_epochs = len(content)
	## Allocate vectors for data
	epoch_ids = np.zeros((n_epochs,1))
	test_accuracy = np.zeros((n_epochs,1))
	for i in range(n_epochs):
		line_i = content[i]
		## Split at all whitespaces
		splitted_line_i = line_i.split()
		epoch_ids[i] = splitted_line_i[1]
		test_accuracy[i] = splitted_line_i[3]
	results = np.concatenate((epoch_ids,test_accuracy),axis=1)
	np.savetxt(out_fname, results, delimiter=',', newline='\n')

