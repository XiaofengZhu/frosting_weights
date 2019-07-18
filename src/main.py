# rm *.txt & ./bash.sh
# experiments/base_model/params.json
# cd /Users/xiaofengzhu/Documents/continual_learning/src
# tensorboard --logdir
import argparse
import logging
import os
import time
import glob
import tensorflow as tf
from model.utils import Params
from model.utils import set_logger
from model.training import train_and_evaluate
from model.reader import load_dataset_from_tfrecords
from model.reader import input_fn
from model.modeling import model_fn
from model.evaluation import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
# loss functions
# cnn, boost, retrain_regu
parser.add_argument('--loss_fn', default='cnn', help="model loss function")
# tf data folder for
# mnist-aug
parser.add_argument('--data_dir', default='../data/mnist-aug',
                    help="Directory containing the dataset")
# test.tfrecords
parser.add_argument('--tfrecords_filename', default='.tfrecords',
                    help="Dataset-filename for the tfrecords")
# usage: python main.py --restore_dir experiments/base_model/best_weights
parser.add_argument('--restore_dir', default=None, # experimens/base_model/best_weights
                    help="Optional, directory containing weights to reload")
# parser.add_argument('--train_range', default='[1-5]',
#                     help="training tf range")
# using pretrained weights and gradient boosting on datasets A and B
# params.num_learners > 1
# parser.add_argument('--retrain', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), \
#     help="try on augmented test dataset")
# train on datasets A and B
parser.add_argument('--combine', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), \
    help="try on augmented test dataset")
parser.add_argument('--finetune', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), \
    help="try on augmented test dataset")
parser.add_argument('--use_kfac', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), \
    help="usek fac true gradient")
parser.add_argument('--log', default='',
                    help="train log postfix")

if __name__ == '__main__':
    # Train the model  
    tf.reset_default_graph()
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)
    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train_{}.log'.format(args.log)))    
    params = Params(json_path)
    if params.mlp_sizes is None or len(params.mlp_sizes) == 0:
        logging.error('mlp_sizes are not set correctly, at least one MLP layer is required')
    params.dict['loss_fn'] = args.loss_fn
    if params.loss_fn == 'boost' and params.num_learners <= 1:
        params.dict['num_learners'] = 2
    params.dict['finetune'] = args.finetune
    params.dict['use_kfac'] = args.use_kfac
    # Load the parameters from the dataset, that gives the size etc. into params
    json_path = os.path.join(args.data_dir, 'dataset_params.json')
    assert os.path.isfile(json_path), "No json file found at {}, please run prepare_data.py".format(json_path)
    params.update(json_path)
    # print(params.dict)
    last_global_epoch, global_epoch = 0, 0
    if params.num_learners <= 1:# not args.retrain or args.combine:
        if args.combine:
            # train from scratch, union of sets
            params.dict['train_size'] = params.dict['train_size'] * 2
            params.dict['vali_size'] = params.dict['vali_size'] * 2
            path_train_tfrecords = os.path.join(args.data_dir, 'train*' + args.tfrecords_filename)
            path_eval_tfrecords = os.path.join(args.data_dir, 'validation*' + args.tfrecords_filename)
            # Create the input data pipeline
            logging.info("Creating the datasets...")
            train_dataset = load_dataset_from_tfrecords(glob.glob(path_train_tfrecords))
            eval_dataset = load_dataset_from_tfrecords(glob.glob(path_eval_tfrecords))
        elif args.finetune:
            args.restore_dir = 'best_weights'
            # path_train_tfrecords = os.path.join(args.data_dir, 'train_aug-*' + args.tfrecords_filename)
            # path_eval_tfrecords = os.path.join(args.data_dir, 'validation_aug' + args.tfrecords_filename)
            # # Create the input data pipeline
            # logging.info("Creating the datasets...")
            # train_dataset = load_dataset_from_tfrecords(glob.glob(path_train_tfrecords))
            # eval_dataset = load_dataset_from_tfrecords(path_eval_tfrecords)
            params.dict['train_size'] = params.dict['train_size'] * 2
            params.dict['vali_size'] = params.dict['vali_size'] * 2
            path_train_tfrecords = os.path.join(args.data_dir, 'train*' + args.tfrecords_filename)
            path_eval_tfrecords = os.path.join(args.data_dir, 'validation*' + args.tfrecords_filename)
            # Create the input data pipeline
            logging.info("Creating the datasets...")
            train_dataset = load_dataset_from_tfrecords(glob.glob(path_train_tfrecords))
            eval_dataset = load_dataset_from_tfrecords(glob.glob(path_eval_tfrecords))            
        else:
            # train-[0-9]
            path_train_tfrecords = os.path.join(args.data_dir, 'train-*'+ args.tfrecords_filename)
            path_eval_tfrecords = os.path.join(args.data_dir, 'validation' + args.tfrecords_filename)
            print(path_train_tfrecords)
            # Create the input data pipeline
            logging.info("Creating the datasets...")
            train_dataset = load_dataset_from_tfrecords(glob.glob(path_train_tfrecords))
            eval_dataset = load_dataset_from_tfrecords(path_eval_tfrecords)
        # Specify other parameters for the dataset and the model
        # Create the two iterators over the two datasets
        train_inputs = input_fn('train', train_dataset, params)
        eval_inputs = input_fn('vali', eval_dataset, params)
        logging.info("- done.")
        # Define the models (2 different set of nodes that share weights for train and validation)
        logging.info("Creating the model...")
        train_model_spec = model_fn('train', train_inputs, params)
        eval_model_spec = model_fn('vali', eval_inputs, params, reuse=True)
        logging.info("- done.")

        logging.info("Starting training for at most {} epoch(s) for the initial learner".format(params.num_epochs))
        start_time = time.time()
        global_epoch = train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, \
            learner_id=0, restore_from=args.restore_dir)
        logging.info("global_epoch: {} epoch(s) at learner 0".format(global_epoch))
        logging.info("total time: %s seconds ---" % (time.time() - start_time))
        # start gradient boosting
        last_global_epoch = global_epoch
    if (params.num_learners > 1):
        #########################################################
        logging.info("RETRAINING ~~")
        params.dict['training_keep_prob'] = 1.0
        start_time = time.time()
        for learner_id in range(1, params.num_learners):
            # # tf.reset_default_graph()
            # # tf.set_random_seed(230)
            # path_train_tfrecords = os.path.join(args.data_dir, 'train_aug-*' + args.tfrecords_filename)
            # path_eval_tfrecords = os.path.join(args.data_dir, 'validation_aug' + args.tfrecords_filename)
            # # Create the input data pipeline
            # logging.info("Creating the datasets...")
            # train_dataset = load_dataset_from_tfrecords(glob.glob(path_train_tfrecords))
            # eval_dataset = load_dataset_from_tfrecords(path_eval_tfrecords)
            # params.dict['train_size'] = params.dict['train_size'] * 2
            # params.dict['vali_size'] = params.dict['vali_size'] * 2
            path_train_tfrecords = os.path.join(args.data_dir, 'train*' + args.tfrecords_filename)
            path_eval_tfrecords = os.path.join(args.data_dir, 'validation*' + args.tfrecords_filename)
            # Create the input data pipeline
            logging.info("Creating the datasets...")
            train_dataset = load_dataset_from_tfrecords(glob.glob(path_train_tfrecords))
            eval_dataset = load_dataset_from_tfrecords(glob.glob(path_eval_tfrecords))            
            # Specify other parameters for the dataset and the model
            # Create the two iterators over the two datasets
            train_inputs = input_fn('train', train_dataset, params)
            eval_inputs = input_fn('vali', eval_dataset, params)
            # logging.info("- done.")
            #########################################################
            logging.info("Creating residual learner ".format(learner_id))
            if params.loss_fn != 'retrain_regu':
                params.dict['use_residual'] = True
            ###END TO DO
            residual_train_model_spec = model_fn('train', train_inputs, params, reuse=tf.AUTO_REUSE, \
                weak_learner_id=learner_id)
            residual_eval_model_spec = model_fn('vali', eval_inputs, params, reuse=True, \
                weak_learner_id=learner_id)
            logging.info("- done.")
            args.restore_dir = 'best_weights'
            global_epoch = train_and_evaluate(residual_train_model_spec, residual_eval_model_spec,
                args.model_dir, params, learner_id=learner_id, restore_from=args.restore_dir, \
                global_epoch=global_epoch)
            logging.info("global_epoch: {} epoch(s) at learner {}".format(global_epoch, learner_id))
            if global_epoch - last_global_epoch == params.early_stoping_epochs:
                logging.info("boosting has stopped early at learner {}".format(learner_id))
                break
            last_global_epoch = global_epoch
        logging.info("total time: %s seconds using boosting ---" % (time.time() - start_time))
