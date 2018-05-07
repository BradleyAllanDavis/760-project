# CapsNet for Landmark Dataset using Dynamic Routing

To train the model, use `python main.py`
You might to specify `--data_path` as command line argument if your dataset directory is not in this same directory.

The accuracy is writen to file "acc_0410.txt"
For landmark data, it needs train data trunks and test data trunks. But I suggest use 9 of train data trunk and 
the last chunk of train data as test data, since it contains 50 classes.

The test accuracy v.s. epochs summary is in `summary.txt`.

The retrieval result (accuracy and pictures) are in folder `./retrieval`.

To retreive, you can use `encode_chunk.py` and `zz_retrieve.py` to first encode the images and then retrieval. Results are shown in folder ./retreival.

For the landmark dataset retrieval result, it is saved in `retrieval_accu_label.txt`. The top-5 labels are stored in order.
The "True label" in .txt file means the query label, and the images in `./retrieval` folder are in format "querylabel_i.jpg".
