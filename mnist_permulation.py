import tensorflow as tf
import numpy as np
from copy import deepcopy
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# read dataset
mnist=read_data_sets("MNIST_data/",one_hot=True)
# return a new mnist dataset w/ pixels randomly permuted
def permute_mnist(mnist,seed):
    perm_inds = range(mnist.train.images.shape[1])
    np.random.seed(seed)
    np.random.shuffle(perm_inds)
    mnist2 = deepcopy(mnist)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(mnist2, set_name) # shallow copy
        this_set._images = np.transpose(np.array([this_set.images[:,c] for c in perm_inds]))
    return mnist2