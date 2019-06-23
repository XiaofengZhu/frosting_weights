# python model/reader.py
import os
import argparse
import glob
import numpy as np
import tensorflow as tf
import argparse
import logging

from model.utils import Params

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='../data/mnist',
                    help="Directory containing the dataset")

def _parse_function(record):
    features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),          
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
    }
    parsed_record = tf.parse_single_example(record, features)
    image = tf.decode_raw(parsed_record['image_raw'], tf.float32)
    label = tf.cast(parsed_record['label'], tf.int32)
    # height = tf.cast(parsed_record['height'], tf.int32)
    # width = tf.cast(parsed_record['width'], tf.int32)
    # depth = tf.cast(parsed_record['depth'], tf.int32)            
    # return image, label, height, width, depth
    return image, label    

def load_dataset_from_tfrecords(path_tfrecords_filename):
    # tfrecords_filename
    # file_type + "_" + tfrecords_filename
    dataset = tf.data.TFRecordDataset(path_tfrecords_filename)   
    # Parse the record into tensors.
    dataset = dataset.map(_parse_function)
    return dataset 
 
def input_fn(mode, dataset, params):
    # Shuffle the dataset
    is_training = (mode == 'train')
    buffer_size = params.buffer_size if is_training else 1
    if mode != 'test':
        dataset = dataset.shuffle(buffer_size=buffer_size)
    batch_size = params.batch_size
    dataset = dataset.batch(batch_size)
    # Repeat the input ## num_epochs times
    dataset = dataset.repeat() 
    # prefetch a batch
    dataset = dataset.prefetch(batch_size)
    # Difference between make_one_shot_iterator between make_initializable_iterator
    # A "one-shot" iterator does not support re-initialization.    
    # The returned iterator will be initialized automatically. 
    iterator = dataset.make_one_shot_iterator()
    # iterator = dataset.make_initializable_iterator()
    features, labels = iterator.get_next()
    # features, labels, width, height, depth = iterator.get_next()
    width = int(params.height)
    height = int(params.width)
    depth = int(params.depth)
    features = tf.reshape(features, [batch_size, \
        height, width, depth])
    # labels = tf.reshape(labels, [-1, 1])
    labels = tf.one_hot(labels, params.num_classes) 
    # iterator_init_op = iterator.initializer
    inputs = {
        # 'iterator_init_op': iterator_init_op,
        'features': features,
        'labels': labels,
        }      
    return inputs

if __name__ == "__main__":
    tf.set_random_seed(230)
    args = parser.parse_args()
    dataset_files = os.path.join(args.data_dir, 'train-*.tfrecords')
    dataset = load_dataset_from_tfrecords(glob.glob(dataset_files))
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    # Load the parameters from the dataset, that gives the size etc. into params
    json_path = os.path.join(args.data_dir, 'dataset_params.json')
    assert os.path.isfile(json_path), "No json file found at {}, please run mnist-to-tfrecords.py".format(json_path)
    params.update(json_path)
    mode = 'train'
    inputs = input_fn(mode, dataset, params)
    # iterator_init_op = inputs['iterator_init_op']
    features, labels = inputs['features'], inputs['labels']
    logging.info("- done loading dataset.")
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # sess.run(iterator_init_op)
        for i in range(4):
            try:
                print(sess.run([tf.shape(features), tf.shape(labels)]))
            except tf.errors.OutOfRangeError:
                print('Done training for {} epochs.', i)
