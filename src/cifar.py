'''
python cifar.py --dataset-name cifar-10
python cifar.py --dataset-name cifar-100
'''
import pickle
import numpy as np
import os
from urllib.request import urlretrieve
import tarfile
import zipfile
import sys
import argparse

def get_data_set(cifar_parent_directory, dataset_name='cifar-10', name="train"):
    if dataset_name=='cifar-10':
        return get_cifar_10_data_set(cifar_parent_directory, name=name)
    else:
        return get_cifar_100_data_set(cifar_parent_directory, name=name)

def get_cifar_10_data_set(cifar_parent_directory, dataset_name='cifar-10', name="train"):
    x = None
    y = None

    data_path = os.path.join(cifar_parent_directory, dataset_name)
    if name is "train":
        for i in range(4):
            f = open(data_path +'/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            _X = np.float32(_X / 255.0)
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            # _X = _X.reshape(-1, 32*32*3)

            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)
    if name is "validation":
        f = open(data_path +'/data_batch_' + str(5), 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        _X = datadict["data"]
        _Y = datadict['labels']

        _X = np.float32(_X / 255.0)
        _X = _X.reshape([-1, 3, 32, 32])
        _X = _X.transpose([0, 2, 3, 1])
        # _X = _X.reshape(-1, 32*32*3)

        if x is None:
            x = _X
            y = _Y
        else:
            x = np.concatenate((x, _X), axis=0)
            y = np.concatenate((y, _Y), axis=0)
    elif name is "test":
        f = open(data_path + '/test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.float32(x / 255.0)
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        # x = x.reshape(-1, 32*32*3)
    # print('*********************', x.shape)
    return {'images': x, 'labels': y}

def get_cifar_100_data_set(cifar_parent_directory, dataset_name='cifar-100', name="train"):
    x = None
    y = None

    data_path = os.path.join(cifar_parent_directory, dataset_name)
    if name is "train":
        f = open(data_path +'/train', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()
        # ['filenames', 'batch_label', 'fine_labels', 'coarse_labels', 'data']
        _X = datadict["data"]
        _Y = datadict['fine_labels']

        _X = np.float32(_X / 255.0)
        _X = _X.reshape([-1, 3, 32, 32])
        _X = _X.transpose([0, 2, 3, 1])
        cut_loc = int(4 * _X.shape[0] / 5)
        x, y = _X[0: cut_loc], _Y[0: cut_loc]

    if name is "validation":
        f = open(data_path +'/train', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        _X = datadict["data"]
        _Y = datadict['fine_labels']

        _X = np.float32(_X / 255.0)
        _X = _X.reshape([-1, 3, 32, 32])
        _X = _X.transpose([0, 2, 3, 1])
        cut_loc = int(4 * _X.shape[0] / 5)
        x, y = _X[cut_loc: ], _Y[cut_loc: ]

    elif name is "test":
        f = open(data_path + '/test', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        x = datadict["data"]
        y = np.array(datadict['fine_labels'])

        x = np.float32(x / 255.0)
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
    # print('*********************', x.shape)
    return {'images': x, 'labels': y}

def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


def _print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()


def maybe_download_and_extract(cifar_parent_directory, dataset_name='cifar-10'):
    cifar_directory = os.path.join(cifar_parent_directory, dataset_name)
    if not os.path.exists(cifar_directory):

        url = "http://www.cs.toronto.edu/~kriz/{}-python.tar.gz".format(dataset_name)
        filename = url.split('/')[-1]
        file_path = os.path.join(cifar_parent_directory, filename)
        zip_cifar_10 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(cifar_parent_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(cifar_parent_directory)
        print("Done.")
        if dataset_name == 'cifar-100':
            tmp_path = os.path.join(cifar_parent_directory, \
            "{}-python".format(dataset_name))
        else:
            tmp_path = os.path.join(cifar_parent_directory, \
            "{}-batches-py".format(dataset_name))           
        os.rename(tmp_path, cifar_directory)
        os.remove(zip_cifar_10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-name', 
        default='cifar-10',
        help='Directory where TFRecords will be stored')
    args = parser.parse_args()
    main_directory = "/tmp/tensorflow"
    maybe_download_and_extract(main_directory, args.dataset_name)