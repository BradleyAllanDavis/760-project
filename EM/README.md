# 760-project

## Setup

The code in this repository is written for python 3. In python 3 the python image library (PIL) is called pillow.

```
	$ conda create -n py3env python=3.5
	$ source activate py3env
	$ conda install numpy
 	$ conda install pillow
 	$ conda install -c anaconda scikit-image
 	$ conda install -c anaconda tensorflow-gpu=1.3.0
 	$ pip install daiquiri
```

The package daiquiri is needed for logging. Pillow is needed to load the landmark dataset. Scikit-image is used in 
generating chunks of the data.

### MNIST Setup

After downloading all the dependencies above, simply run

```
    $ source activate py3env
	$ chmod +x downloadMNIST.sh
	$ ./downloadMNIST.sh && cd ..
	$ python trainMNIST.py 
```

Test

```
	$ python testMNISt.py 
```

### Landmark

Need landmark chunks inside `data/landmark_chunks` to run.

### Running MNIST vs Landmark

MNIST and Landmark share config file.
To run MNIST, uncomment data/mnist as the path and comment out data/landmark path.
