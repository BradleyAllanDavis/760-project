import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--data_path', default="../", type=str)
parser.add_argument('--metadata_path',default="../train.npy",type=str)
parser.add_argument('--CheckPointPath', default="./model/my_model.ckpt", type=str)

parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--height", default=28, type=int)
parser.add_argument("--width", default=28, type=int)
parser.add_argument("--num_channel", default=3, type=int)
parser.add_argument("--num_class", default=50, type=int)
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("--stddev", default=0.1, type=float)
parser.add_argument("--epsilon", default=1e-9, type=float)

args = parser.parse_args()
