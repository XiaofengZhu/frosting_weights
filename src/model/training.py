"""Tensorflow utility functions for training"""
# tensorboard --logdir=experiments/base_model/
# tensorboard --logdir=experiments/base_model/train_summaries
# tensorboard --logdir=experiments/base_model/eval_summaries

import logging
import os

from tqdm import trange
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

from model.utils import save_dict_to_json, load_best_metric, get_expaned_metrics
from model.evaluation import evaluate_sess
from model.modeling import retrain_lenet, get_residual
import tensorflow.contrib.slim as slim

def train_sess(sess, model_spec, num_steps, writer, params):
    """Train the model on `num_steps` batches

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries
        params: (Params) hyperparameters
    """
    # Get relevant graph operations or nodes needed for training

    loss = model_spec['loss']
    train_op = model_spec['train_op']
    update_metrics = model_spec['update_metrics']
    metrics = model_spec['metrics']
    summary_op = model_spec['summary_op']
    global_step = tf.train.get_global_step()

    # Load the training dataset into the pipeline and initialize the metrics local variables
    # sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])
    # Use tqdm for progress bar
    t = trange(int(num_steps))
    for i in t:
        # Evaluate summaries for tensorboard only once in a while
        if i == params.save_summary_steps - 1:
        # if i % params.save_summary_steps == 0:
            # Perform a mini-batch update
            _, _, loss_val, summ, global_step_val = sess.run([train_op, update_metrics, loss,
                                                              summary_op, global_step])
            # Write summaries for tensorboard
            writer.add_summary(summ, global_step_val)
        else:
            _, _, loss_val = sess.run([train_op, update_metrics, loss])
        # Log the loss in the tqdm progress bar
        # t.set_postfix(loss='{:05.3f}'.format(loss_val))
    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    expanded_metrics_val = get_expaned_metrics(metrics_val)
    metrics_string = " ; ".join("{}: {:05.4f}".format(k, v) for k, v in expanded_metrics_val.items())
    logging.info("- Train metrics: " + metrics_string)

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def isSavingWeights(eval_metrics, best_eval_metrics):
    for i in range(len(eval_metrics)):
        if eval_metrics[i] > best_eval_metrics[i]:
            return True
        elif eval_metrics[i] < best_eval_metrics[i]:
            return False
        else:
            continue
    return False

def train_and_evaluate(train_model_spec, eval_model_spec,
    model_dir, params, learner_id=0, restore_from=None, global_epoch=1):
    """Train the model and evaluate every epoch.

    Args:
        train_model_spec: (dict) contains the graph operations or nodes needed for training
        eval_model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # Initialize tf.Saver instances to save weights during training
    last_saver = tf.train.Saver() # will keep last 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1)  # only keep 1 best checkpoint (best on eval)
    begin_at_epoch = 0
    with tf.Session() as sess:
        # Initialize model variables
        tf.reset_default_graph()
        sess.run(train_model_spec['variable_init_op'])
        # For tensorboard (takes care of writing summaries to files)
        train_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train_summaries'), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(model_dir, 'vali_summaries'), sess.graph)
        best_json_path = os.path.join(model_dir, "metrics_eval_best_weights.json")
        best_eval_metrics = [0.0, -float('inf')]
        # global_epoch = 0
        # Reload weights from directory if specified
        # restor from the previous learner
        if restore_from is not None:
            save_path = os.path.join(model_dir, restore_from)
            if os.path.isdir(save_path):
                save_path = tf.train.latest_checkpoint(save_path)
                begin_at_epoch = int(save_path.split('-')[-1])
                global_epoch = begin_at_epoch       
            logging.info("Restoring parameters from {}".format(save_path))
            # last_saver = tf.train.import_meta_graph(save_path+".meta")
            if params.loss_fn == 'retrain_regu_mine':
                pretrained_include = ['model/cnn']
            elif params.loss_fn == 'cnn' and params.finetune:
                pretrained_include = ['model/cnn']
            else:
                pretrained_include = ['model/c_cnn']
                pretrained_include.append('model/cnn')
            # if params.loss_fn=='boost':
            #     pretrained_include = ['model/boost']
            # for i in range(1, learner_id):
            #     pretrained_include.append('residual_mlp_{}'.format(learner_id))
            pretrained_vars = tf.contrib.framework.get_variables_to_restore(include=pretrained_include)
            pretrained_saver = tf.train.Saver(pretrained_vars, name="pretrained_saver")
            pretrained_saver.restore(sess, save_path)
            # if params.num_learners > 1:
            #     best_eval_metrics = load_best_metric(best_json_path)
            #     best_eval_metrics = [best_eval_metrics['accuracy'], -best_eval_metrics['loss']]
        model_summary()
        # for each learner
        early_stopping_count = 0
        for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs):
            if early_stopping_count == int(params.early_stoping_epochs):
                logging.info("Early stopping at learner {}, epoch {}/{}".format(learner_id, epoch + 1, \
                    begin_at_epoch + params.num_epochs))
                break
            # Run one epoch
            logging.info("Learner {}, Epoch {}/{}".format(learner_id, epoch + 1, \
                begin_at_epoch + params.num_epochs))
            # logging.info(global_epoch)
            # Compute number of batches in one epoch (one full pass over the training set)
            num_steps = (params.train_size + params.batch_size - 1) // params.batch_size
            train_sess(sess, train_model_spec, num_steps, train_writer, params)
            # Save weights
            last_save_path = os.path.join(model_dir, 'last_weights', 'after-epoch')
            # global_epoch = int(params.num_learners) * int(params.num_epochs) + epoch + 1
            last_saver.save(sess, last_save_path, global_step=global_epoch)
            # Evaluate for one epoch on validation set
            num_steps = (params.vali_size + params.batch_size - 1) // params.batch_size
            metrics = evaluate_sess(sess, eval_model_spec, num_steps, eval_writer, params)
            # If best_eval, best_save_path
            accuracy_metric = round(metrics['accuracy'], 6)
            loss_metric = -round(metrics['loss'], 6)
            # save_batch()
            eval_metrics = [accuracy_metric, loss_metric]
            # logging.info('global_epoch: {}, best_eval_metrics: {}, \
            #     eval_metric: {}', global_epoch, best_eval_metrics, eval_metric)
            if isSavingWeights(eval_metrics, best_eval_metrics):
                # rest early_stopping_count
                early_stopping_count = 0
                # and isSavingWeights
                best_eval_metrics = eval_metrics
                # Save weights
                # trainalbe_vars = {v.name: v for v in tf.trainable_variables() if 'model' in v.name}
                # print(trainalbe_vars.keys())                    
                if params.loss_fn == 'cnn' or params.loss_fn == 'retrain_regu':
                    cnn_vars=[v for v in tf.trainable_variables() if 'model/cnn' in v.name]
                    # c_cnn_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/c_cnn')
                    c_cnn_vars=[v for v in tf.trainable_variables() if 'model/c_cnn' in v.name]
                    update_weights = [tf.assign(c, old) for (c, old) in \
                    zip(c_cnn_vars, cnn_vars)]
                    sess.run(update_weights)
                '''
                if params.loss_fn == 'boost':

                    cnn_vars=[v for v in tf.trainable_variables() if 'model/cnn' in v.name]
                    c_cnn_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/c_cnn')
                    update_weights = [tf.assign(c, old) for (c, old) in \
                    zip(c_cnn_vars, cnn_vars)]
                    sess.run(update_weights)

                    features = train_model_spec['features']
                    labels = train_model_spec['labels']
                    predicted_scores, _ = retrain_lenet(features, params, var_scope='model/c_cnn')
                    residuals = get_residual(labels, predicted_scores)
                    train_model_spec['old_predicted_scores'] = predicted_scores
                    train_model_spec['residuals'] = residuals

                    features = eval_model_spec['features']
                    labels = eval_model_spec['labels']
                    predicted_scores, _ = retrain_lenet(features, params, var_scope='model/c_cnn')
                    residuals = get_residual(labels, predicted_scores)
                    eval_model_spec['old_predicted_scores'] = predicted_scores
                    eval_model_spec['residuals'] = residuals
                    
                    sess.run(train_model_spec['old_predicted_scores'])
                    sess.run(train_model_spec['residuals'])

                    sess.run(eval_model_spec['old_predicted_scores'])
                    sess.run(eval_model_spec['residuals'])
                '''
                best_save_path = os.path.join(model_dir, 'best_weights', 'after-epoch')
                # global_epoch = int(params.num_learners) * int(params.num_epochs) + epoch + 1
                best_save_path = best_saver.save(sess, best_save_path, global_step=global_epoch)
                logging.info("- Found new best metric score, saving in {}".format(best_save_path))
                # Save best eval metrics in a json file in the model directory
                save_dict_to_json(metrics, best_json_path)
                save_dict_to_json({'stopped_at_learner': learner_id}, \
                    os.path.join(model_dir, 'best_weights', 'learner.json'))
            else:
                early_stopping_count = early_stopping_count + 1
            # Save latest eval metrics in a json file in the model directory
            last_json_path = os.path.join(model_dir, "metrics_eval_last_weights.json")
            save_dict_to_json(metrics, last_json_path)
            global_epoch += 1
    return global_epoch
