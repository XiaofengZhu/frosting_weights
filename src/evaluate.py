"""Evaluate the model"""

import argparse
import logging
import os

import numpy as np
import tensorflow as tf

from model.utils import Params
from model.utils import set_logger, load_best_metric, load_learner_id
from model.evaluation import evaluate
from model.reader import input_fn
from model.reader import load_dataset_from_tfrecords
from model.modeling import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--residual_model_dir', default='experiments/residual_model',
                    help="Directory containing params.json")
# loss functions
# cnn, boost, retrain_regu
parser.add_argument('--loss_fn', default='cnn', help="model loss function")
# tf data folder for
# mnist
parser.add_argument('--data_dir', default='../data/mnist-aug',
                    help="Directory containing the dataset")
# test.tfrecords
parser.add_argument('--tfrecords_filename', default='.tfrecords',
                    help="Dataset-filename for the tfrecords")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of the best weights")
parser.add_argument('--aug', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), \
    help="test on augmented test dataset")
parser.add_argument('--combine', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), \
    help="test on old and augmented test datasets")
parser.add_argument('--finetune', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), \
    help="finetune mode")
parser.add_argument('--log', default='',
                    help="test log postfix")

if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    if params.mlp_sizes is None or len(params.mlp_sizes) == 0:
        logging.error('mlp_sizes are not set correctly, at least one MLP layer is required')
    params.dict['loss_fn'] = args.loss_fn
    params.dict['finetune'] = args.finetune    
    params.dict['training_keep_prob'] = 1.0
    if params.loss_fn == 'boost' and params.num_learners <= 1:
        params.dict['num_learners'] = 2
    if params.num_learners > 1 and params.loss_fn != 'retrain_regu':
        params.dict['use_residual'] = True
    # Load the parameters from the dataset, that gives the size etc. into params
    json_path = os.path.join(args.data_dir, 'dataset_params.json')
    assert os.path.isfile(json_path), "No json file found at {}, run build.py".format(json_path)
    params.update(json_path)
    # Set the logger
    set_logger(os.path.join(args.model_dir, 'test_{}.log'.format(args.log)))
    # # Get paths for tfrecords
    dataset = 'test'
    if args.aug:
        print('USING augmented TEST')
        dataset += '_aug'
    if args.combine:
        print('USING both Tests')
        dataset += '*'        
    path_eval_tfrecords = os.path.join(args.data_dir, dataset + args.tfrecords_filename)
    # Create the input data pipeline
    logging.info("Creating the dataset...")
    eval_dataset = load_dataset_from_tfrecords(path_eval_tfrecords)
    # Create iterator over the test set
    eval_inputs = input_fn('test', eval_dataset, params)
    logging.info("- done.")
    # Define the model
    logging.info("Creating the model...")
    weak_learner_id = load_learner_id(os.path.join(args.model_dir, args.restore_from, 'learner.json'))[0]
    eval_model_spec = model_fn('test', eval_inputs, params, reuse=False, \
        weak_learner_id=int(weak_learner_id))
    # node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    # print(node_names)
    logging.info("- done.")
    logging.info("Starting evaluation")
    logging.info("Optimized using {} learners".format(weak_learner_id))
    evaluate(eval_model_spec, args.model_dir, params, args.restore_from)
