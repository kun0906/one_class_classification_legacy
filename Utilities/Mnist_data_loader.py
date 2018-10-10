"""
    to be implement

"""

import gzip
import numpy as np


class MNIST_DataLoader():
    def __init__(self, ad_experiment=1):
        pass


def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)

    return data
