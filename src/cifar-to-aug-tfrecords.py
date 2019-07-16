#! /usr/env/bin python3

"""
Convert CIFAR Dataset to local TFRecords
python cifar-to-aug-tfrecords.py --data-directory ../data/cifar-10 --dataset-name cifar-10
python cifar-to-aug-tfrecords.py --data-directory ../data/cifar-100 --dataset-name cifar-100
"""

import argparse
import os
import sys
import logging
import tensorflow as tf
from cifar import get_data_set, maybe_download_and_extract

from model.utils import save_dict_to_json
from aug_images import augment_data

def _data_path(data_directory:str, name:str) -> str:
    """Construct a full path to a TFRecord file to be stored in the 
    data_directory. Will also ensure the data directory exists
    
    Args:
        data_directory: The directory where the records will be stored
        name:           The name of the TFRecord
    
    Returns:
        The full path to the TFRecord file
    """
    if not os.path.isdir(data_directory):
        os.makedirs(data_directory)

    return os.path.join(data_directory, f'{name}.tfrecords')

def _int64_feature(value:int) -> tf.train.Features.FeatureEntry:
    """Create a Int64List Feature
    
    Args:
        value: The value to store in the feature
    
    Returns:
        The FeatureEntry
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value:str) -> tf.train.Features.FeatureEntry:
    """Create a BytesList Feature
    
    Args:
        value: The value to store in the feature
    
    Returns:
        The FeatureEntry
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(data_set, name:str, data_directory:str, num_shards:int=1, aug:bool=False):
    """Convert the dataset into TFRecords on disk
    
    Args:
        data_set:       The MNIST data set to convert
        name:           The name of the data set
        data_directory: The directory where records will be stored
        num_shards:     The number of files on disk to separate records into
    """
    print(f'\nProcessing {name} data')

    images = data_set['images']
    labels = data_set['labels']
    if aug:
        images, labels = augment_data(images, labels)    
    # logging.warning('*********************', images.shape)
    num_examples, rows, cols, depth = images.shape

    def _process_examples(start_idx:int, end_index:int, filename:str):
        with tf.python_io.TFRecordWriter(filename) as writer:
            for index in range(start_idx, end_index):
                sys.stdout.write(f"\rProcessing sample {index+1} of {num_examples}")
                sys.stdout.flush()

                image_raw = images[index].tostring()
                
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(rows),
                    'width': _int64_feature(cols),
                    'depth': _int64_feature(depth),
                    'label': _int64_feature(int(labels[index])),
                    'image_raw': _bytes_feature(image_raw)
                }))
                writer.write(example.SerializeToString())
    
    if num_shards == 1:
        _process_examples(0, num_examples, _data_path(data_directory, name))
    else:
        total_examples = num_examples
        samples_per_shard = total_examples // num_shards

        for shard in range(num_shards):
            start_index = shard * samples_per_shard
            end_index = start_index + samples_per_shard
            _process_examples(start_index, end_index, _data_path(data_directory, f'{name}-{shard+1}'))

    return num_examples, rows, cols, depth

def convert_to_tf_record(data_directory:str, dataset_name:str):
    """Convert the TF MNIST Dataset to TFRecord formats
    
    Args:
        data_directory: The directory where the TFRecord files should be stored
    """
    dataset_parent_path = "/tmp/tensorflow"
    maybe_download_and_extract(dataset_parent_path, dataset_name)
    cifar10_train = get_data_set(dataset_parent_path, dataset_name, 'train')
    cifar10_validation = get_data_set(dataset_parent_path, dataset_name, 'validation')
    cifar10_test = get_data_set(dataset_parent_path, dataset_name, 'test')
    
    num_validation_examples, rows, cols, depth = convert_to(cifar10_validation, 'validation', data_directory)
    num_validation_aug_examples, rows, cols, depth = convert_to(cifar10_validation, 'validation_aug', data_directory, aug=True)   
    num_train_examples, rows, cols, depth = convert_to(cifar10_train, 'train', data_directory, num_shards=10)
    num_train_aug_examples, rows, cols, depth = convert_to(cifar10_train, 'train_aug', data_directory, num_shards=10, aug=True)    
    num_test_examples, rows, cols, depth = convert_to(cifar10_test, 'test', data_directory)
    num_test_aug_examples, rows, cols, depth = convert_to(cifar10_test, 'test_aug', data_directory, aug=True)
    # Save datasets properties in json file
    sizes = {
        'height': rows,
        'width': cols,
        'depth': depth,
        'vali_size': num_validation_aug_examples,
        # 'vali_aug_size': num_validation_aug_examples,
        'train_size': num_train_aug_examples,
        # 'train_aug_size': num_train_aug_examples,
        'test_size': num_test_aug_examples,
        # 'test_aug_size': num_test_aug_examples
    }
    save_dict_to_json(sizes, os.path.join(data_directory, 'dataset_params.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--data-directory', 
        default='../data/cifar-10-aug',
        help='Directory where TFRecords will be stored')
    parser.add_argument(
        '--dataset-name', 
        default='cifar-10',
        help='Directory where TFRecords will be stored')
    args = parser.parse_args()
    convert_to_tf_record(os.path.expanduser(args.data_directory), args.dataset_name)
 