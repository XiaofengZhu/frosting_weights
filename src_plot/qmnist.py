import tensorflow as tf
import numpy as np
import os
import sys
import struct
import matplotlib.pyplot as plt

RAW_QMNIST_DATA = os.environ.get('RAW_QMNIST_DATA')

def load_mnist(path, kind='test'):
    labels_path = os.path.join(path,
                               'qmnist-%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               'qmnist-%s-images-idx3-ubyte.gz'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    # with open(images_path, 'rb') as imgpath:
    #     magic, num, rows, cols = struct.unpack('>IIII',
    #                                            imgpath.read(16))
    #     images = np.fromfile(imgpath,
    #                          dtype=np.uint8).reshape(len(labels), 784)

    return labels, labels

X_test, y_test= load_mnist(RAW_QMNIST_DATA)
print(np.shape(X_test), np.shape(y_test))