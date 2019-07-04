"""Define the model."""
import sys, random, logging
import tensorflow as tf
import numpy as np
from util import loss_fns, search_metrics
from tensorflow.python.ops import array_ops

from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os
import functools
import time

#################
def lenet(X, params=None, var_scope='cnn'):
    with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
        # CONVOLUTION 1 - 1
        with tf.name_scope('conv1_1'):
            filter1_1 = tf.get_variable('weights1_1', shape=[5, 5, int(params.depth), 32], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
            stride = [1,1,1,1]
            conv = tf.nn.conv2d(X, filter1_1, stride, padding='SAME')
            biases = tf.get_variable('biases1_1', shape=[32], \
                initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(out)
        # POOL 1
        with tf.name_scope('pool1'):
            pool1_1 = tf.nn.max_pool(conv1_1,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding='SAME',
                                     name='pool1_1')
            pool1_1_drop = tf.nn.dropout(pool1_1, params.training_keep_prob)
        # CONVOLUTION 1 - 2
        with tf.name_scope('conv1_2'):
            filter1_2 = tf.get_variable('weights1_2', shape=[5, 5, 32, 64], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
            conv = tf.nn.conv2d(pool1_1_drop, filter1_2, [1,1,1,1], padding='SAME')
            biases = tf.get_variable('biases1_2', shape=[64], \
                initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out)
        # POOL 2
        with tf.name_scope('pool2'):
            pool2_1 = tf.nn.max_pool(conv1_2,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding='SAME',
                                     name='pool2_1')
            pool2_1_drop = tf.nn.dropout(pool2_1, params.training_keep_prob)
        #FULLY CONNECTED 1
        with tf.name_scope('fc1') as scope:
            pool2_flat = tf.layers.Flatten()(pool2_1_drop)
            dim = pool2_flat.get_shape()[1].value
            fc1w = tf.get_variable('weights3_1', shape=[dim, 1024], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
            fc1b = tf.get_variable('biases3_1', shape=[1024], \
                initializer=tf.constant_initializer(1.0))
            out = tf.nn.bias_add(tf.matmul(pool2_flat, fc1w), fc1b)
            fc1 = tf.nn.relu(out)
            fc1_drop = tf.nn.dropout(fc1, params.training_keep_prob)
        #FULLY CONNECTED 2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.get_variable('weights3_2', shape=[1024, params.num_classes], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
            fc2b = tf.get_variable('biases3_2', shape=[params.num_classes], \
                initializer=tf.constant_initializer(1.0))
            Ylogits = tf.nn.bias_add(tf.matmul(fc1_drop, fc2w), fc2b)
    return Ylogits, pool2_flat

def retrain_lenet(X, params=None, var_scope='cnn'):
    trainable = var_scope=='cnn'
    neurons = []
    weights = []
    with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
        # CONVOLUTION 1 - 1
        with tf.name_scope('conv1_1'):
            filter1_1 = tf.get_variable('weights1_1', shape=[5, 5, int(params.depth), 32], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1), trainable=trainable)
            stride = [1,1,1,1]
            conv = tf.nn.conv2d(X, filter1_1, stride, padding='SAME')
            biases = tf.get_variable('biases1_1', shape=[32], \
                initializer=tf.constant_initializer(0.0), trainable=trainable)
            out = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(out)
            weights.extend([filter1_1, biases])
            neurons.append(conv1_1)
        # POOL 1
        with tf.name_scope('pool1'):
            pool1_1 = tf.nn.max_pool(conv1_1,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding='SAME',
                                     name='pool1_1')
            pool1_1_drop = tf.nn.dropout(pool1_1, params.training_keep_prob)
        # CONVOLUTION 1 - 2
        with tf.name_scope('conv1_2'):
            filter1_2 = tf.get_variable('weights1_2', shape=[5, 5, 32, 64], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1), trainable=trainable)
            conv = tf.nn.conv2d(pool1_1_drop, filter1_2, [1,1,1,1], padding='SAME')
            biases = tf.get_variable('biases1_2', shape=[64], \
                initializer=tf.constant_initializer(0.0), trainable=trainable)
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out)
            weights.extend([filter1_2, biases])
            neurons.append(conv1_2)
        # POOL 2
        with tf.name_scope('pool2'):
            pool2_1 = tf.nn.max_pool(conv1_2,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding='SAME',
                                     name='pool2_1')
            pool2_1_drop = tf.nn.dropout(pool2_1, params.training_keep_prob)
        #FULLY CONNECTED 1
        with tf.name_scope('fc1') as scope:
            pool2_flat = tf.layers.Flatten()(pool2_1_drop)
            dim = pool2_flat.get_shape()[1].value
            fc1w = tf.get_variable('weights3_1', shape=[dim, 1024], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1), trainable=trainable)
            fc1b = tf.get_variable('biases3_1', shape=[1024], \
                initializer=tf.constant_initializer(1.0), trainable=trainable)
            out = tf.nn.bias_add(tf.matmul(pool2_flat, fc1w), fc1b)
            fc1 = tf.nn.relu(out)
            fc1_drop = tf.nn.dropout(fc1, params.training_keep_prob)
            weights.extend([fc1w, fc1b])
            neurons.append(fc1_drop)
        #FULLY CONNECTED 2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.get_variable('weights3_2', shape=[1024, params.num_classes], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1), trainable=trainable)
            fc2b = tf.get_variable('biases3_2', shape=[params.num_classes], \
                initializer=tf.constant_initializer(1.0), trainable=trainable)
            Ylogits = tf.nn.bias_add(tf.matmul(fc1_drop, fc2w), fc2b)
            weights.extend([fc2w, fc2b])
            neurons.append(Ylogits)
    return Ylogits, (neurons, weights)
'''
def build_residual_model(mode, inputs, params, weak_learner_id):
    """Compute logits of the model (output distribution)
    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, residuals...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
    Returns:
        output: (tf.Tensor) output of the model
    Notice:
        !!! boosting is only supported for cnn and urrank
    """
    is_test = (mode == 'test')
    features = inputs['features']
    # if 'old_predicted_scores' not in inputs:
    #     logging.error('residuals not in inputs')
    #     labels = inputs['labels']
    #     predicted_scores, _ = retrain_lenet(features, params, var_scope='c_cnn')
    #     inputs['old_predicted_scores'] = predicted_scores
    residual_predicted_scores, _ = retrain_lenet(features, params, var_scope='cnn')
    # residual_predicted_scores = tf.Print(residual_predicted_scores, [residual_predicted_scores], \
    #     message='residual_predicted_scores\n')
    # boosted_scores = inputs['old_predicted_scores'] + residual_predicted_scores
    return residual_predicted_scores, None

def build_residual_model(mode, inputs, params, weak_learner_id):
    """Compute logits of the model (output distribution)
    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, residuals...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
    Returns:
        output: (tf.Tensor) output of the model
    Notice:
        !!! boosting is only supported for cnn and urrank
    """
    is_test = (mode == 'test')
    features = inputs['features']
    if 'old_predicted_scores' not in inputs:
        logging.error('residuals not in inputs')
        labels = inputs['labels']
        predicted_scores, _ = retrain_lenet(features, params, var_scope='c_cnn')
        inputs['old_predicted_scores'] = predicted_scores
        residuals = 100 * get_residual(labels, predicted_scores)
        inputs['residuals'] = residuals
    residual_predicted_scores, _ = retrain_lenet(features, params, var_scope='cnn')
    # residual_predicted_scores = tf.Print(residual_predicted_scores, [residual_predicted_scores], \
    #     message='residual_predicted_scores\n')
    boosted_scores = inputs['old_predicted_scores'] + residual_predicted_scores
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                    logits=inputs['old_predicted_scores']+inputs['residuals'])
    calculated_loss = tf.reduce_mean(cross_entropy)
    inputs['calculated_loss'] = calculated_loss
    mse_loss = tf.losses.mean_squared_error(residuals, residual_predicted_scores)
    # inputs['old_predicted_scores']+inputs['residuals']
    return boosted_scores, mse_loss

def build_residual_model(mode, inputs, params, weak_learner_id):
    """Compute logits of the model (output distribution)
    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, residuals...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
    Returns:
        output: (tf.Tensor) output of the model
    Notice:
        !!! boosting is only supported for cnn and urrank
    """
    is_test = (mode == 'test')
    features = inputs['features']
    if 'residuals' not in inputs:
        logging.error('residuals not in inputs')
        labels = inputs['labels']
        predicted_scores, _ = retrain_lenet(features, params, var_scope='c_cnn')
        residuals = get_residual(labels, predicted_scores)
        inputs['old_predicted_scores'] = predicted_scores
        # inputs['old_predicted_scores'] = tf.stop_gradient(inputs['old_predicted_scores'])
        inputs['residuals'] = residuals
        # inputs['residuals'] = tf.stop_gradient(inputs['residuals'])

    residual_predicted_scores, _ = retrain_lenet(features, params, var_scope='cnn')
    # residual_predicted_scores = tf.Print(residual_predicted_scores, [residual_predicted_scores], \
    #     message='residual_predicted_scores\n')
    boosted_scores = inputs['old_predicted_scores'] + residual_predicted_scores
    if is_test:
        return boosted_scores, None
    square_loss = tf.square(inputs['residuals'] - residual_predicted_scores)
    square_loss = tf.reduce_sum(square_loss)
    mse_loss = square_loss/features.get_shape()[1].value
    # is_test = (mode == 'test')
    # boosted_scores = tf.constant(0.0, dtype=tf.float32)
    # # MLP netowork for residuals
    # features = inputs['features']
    # if params.loss_fn == 'boost':
    #     predicted_scores, _ = lenet(features, params, var_scope='c_cnn')
    # else:
    #     logging.error('Loss function not supported for boosting')
    #     sys.exit(1)
    # # only one weak learner for now, weak_learner_id==1
    # # for trained_learner_id in range(1, weak_learner_id):
    # #     n_predicted_scores, _ = lenet(features, params, var_scope='c_cnn'+str(trained_learner_id))
    # #     predicted_scores += n_predicted_scores
    # predicted_scores = tf.stop_gradient(predicted_scores)
    # # predicted_scores = tf.Print(predicted_scores, [predicted_scores], message='predicted_scores\n')
    # residual_predicted_scores, _ = lenet(features, params, var_scope='cnn')
    # # boosted_scores = predicted_scores + 1/math.sqrt(weak_learner_id) * residual_predicted_scores
    # boosted_scores = predicted_scores + residual_predicted_scores
    # if is_test:
    #     return boosted_scores, None
    # labels = inputs['labels']
    # residuals = get_residual(labels, predicted_scores)
    # # residuals = tf.Print(residuals, [residuals], message='residuals\n')
    # # residual_predicted_scores = tf.Print(residual_predicted_scores, [residual_predicted_scores], message='residual_predicted_scores\n')
    # mse_loss = tf.losses.mean_squared_error(residuals, residual_predicted_scores)
    return boosted_scores, mse_loss
'''
# new weights for fc1_drop
# def build_residual_model(is_training, inputs, params, weak_learner_id):
#     """Compute logits of the model (output distribution)
#     Args:
#         mode: (string) 'train', 'eval', etc.
#         inputs: (dict) contains the inputs of the graph (features, residuals...)
#                 this can be `tf.placeholder` or outputs of `tf.data`
#         params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
#     Returns:
#         output: (tf.Tensor) output of the model
#     Notice:
#         !!! boosting is only supported for cnn and urrank
#     """
#     mse_loss = tf.constant(0.0, dtype=tf.float32)
#     # MLP netowork for residuals
#     features = inputs['features']
#     if params.loss_fn == 'boost':
#         predicted_scores, fc1_drop = lenet(features, params, var_scope='c_cnn')
#         fc1_drop = tf.stop_gradient(fc1_drop)
#     else:
#         logging.error('Loss function not supported for boosting')
#         sys.exit(1)
#     if weak_learner_id >= 1:
#         for trained_learner_id in range(1, weak_learner_id):
#             predicted_scores += _get_residual_mlp_logits(fc1_drop, params, \
#             weak_learner_id=trained_learner_id)
#         predicted_scores = tf.stop_gradient(predicted_scores)
#         residual_predicted_scores = _get_residual_mlp_logits(fc1_drop, params, \
#             weak_learner_id=weak_learner_id)
#         # boosted_scores = predicted_scores + 1/math.sqrt(weak_learner_id) * residual_predicted_scores
#         boosted_scores = predicted_scores + residual_predicted_scores
#     else:
#         boosted_scores = predicted_scores
#     if not is_training:
#         return boosted_scores, mse_loss
#     if weak_learner_id >= 1:
#         labels = inputs['labels']
#         residuals = get_residual(labels, predicted_scores)
#         mse_loss = tf.losses.mean_squared_error(residuals, residual_predicted_scores)
#     return boosted_scores, mse_loss


# # new weights for fc1_drop
# def _get_residual_mlp_logits(features, params, weak_learner_id=1):
#     with tf.variable_scope('residual_mlp_{}'.format(weak_learner_id), reuse=tf.AUTO_REUSE):
#         logits = tf.layers.dense(features, params.num_classes,
#             name='residual_{}_dense_{}'.format(weak_learner_id, len(params.residual_mlp_sizes)))
#     return logits

# new weights for pool2_flat
def build_residual_model(mode, inputs, params, weak_learner_id):
    """Compute logits of the model (output distribution)
    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, residuals...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
    Returns:
        output: (tf.Tensor) output of the model
    Notice:
        !!! boosting is only supported for cnn and urrank
    """
    is_training = (mode == 'train')
    is_test = (mode == 'test')
    mse_loss = tf.constant(0.0, dtype=tf.float32)
    # MLP netowork for residuals
    features = inputs['features']
    if params.loss_fn == 'boost':
        predicted_scores, pool2_flat = lenet(features, params)
    else:
        logging.error('Loss function not supported for boosting')
        sys.exit(1)
    if weak_learner_id >= 1:
        for trained_learner_id in range(1, weak_learner_id):
            predicted_scores += _get_residual_mlp_logits(pool2_flat, params, \
            weak_learner_id=trained_learner_id)
        predicted_scores = tf.stop_gradient(predicted_scores)
        residual_predicted_scores = _get_residual_mlp_logits(pool2_flat, params, \
            weak_learner_id=weak_learner_id)
        # boosted_scores = predicted_scores + 1/math.sqrt(weak_learner_id) * residual_predicted_scores
        boosted_scores = predicted_scores + residual_predicted_scores
    else:
        boosted_scores = predicted_scores
    if is_test:
        return boosted_scores, None
    if weak_learner_id >= 1:
        labels = inputs['labels']
        residuals = get_residual(labels, predicted_scores)
        mse_loss = tf.losses.mean_squared_error(residuals, residual_predicted_scores)
    return boosted_scores, mse_loss

# new weights for pool2_flat
def _get_residual_mlp_logits(features, params, weak_learner_id=1):
    with tf.variable_scope('residual_mlp_{}'.format(weak_learner_id), reuse=tf.AUTO_REUSE):
        out = tf.layers.dense(features, params.residual_mlp_sizes[0],
            name='residual_{}_dense_0'.format(weak_learner_id), activation=tf.nn.relu)
        for i in range(1, len(params.residual_mlp_sizes)):
            out = tf.layers.dense(out, params.residual_mlp_sizes[i], \
                name='residual_{}_dense_{}'.format(weak_learner_id, i), activation=tf.nn.relu)
        logits = tf.layers.dense(out, params.num_classes,
            name='residual_{}_dense_{}'.format(weak_learner_id, len(params.residual_mlp_sizes)))
    return logits

def _get_mlp_logits(features, params):
    with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
        out = tf.layers.dense(features, params.mlp_sizes[0], \
            name='dense_0', activation=tf.nn.relu)
        for i in range(1, len(params.mlp_sizes)):
            out = tf.layers.dense(out, params.mlp_sizes[i], \
                name='dense_{}'.format(i), activation=tf.nn.relu)
        logits = tf.layers.dense(out, 1, \
            name='dense_{}'.format(len(params.mlp_sizes)))
    return logits

def get_residual(labels, Ylogits):
    Ysoftmax = tf.nn.softmax(Ylogits)
    return labels - Ysoftmax

def build_model(mode, inputs, params, weak_learner_id):
    """Compute logits of the model
    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
    Returns:
        output: (tf.Tensor) output of the model
    Notice:
        !!! when using the build_model mdprank needs a learning_rate around 1e-5 - 1e-7
    """
    is_training = (mode == 'train')
    is_test = (mode == 'test') 
    features = inputs['features']
    if params.loss_fn=='retrain_regu':
        if not is_test:
            _, (old_neurons, old_weights) = retrain_lenet(features, params, var_scope='c_cnn')
            y_conv, (neurons, weights) = retrain_lenet(features, params, var_scope='cnn')
            neuron_mse_list = [tf.losses.mean_squared_error(old_neuron, neuron) for (old_neuron, neuron) \
            in zip(old_neurons, neurons)]
            neuron_mses = functools.reduce(lambda x,y:x+y, neuron_mse_list)
            # weight regulization
            var_mse_list = [tf.losses.mean_squared_error(old_var, var) for (old_var, var) \
            in zip(old_weights, weights)]
            var_mses = functools.reduce(lambda x,y:x+y, var_mse_list)
            regulization_loss = 0.001 * neuron_mses + 0.001 * var_mses            
            return y_conv, regulization_loss
        return retrain_lenet(features, params, var_scope='cnn')
    if params.use_residual:
        # y_conv, (neurons, weights) = retrain_lenet(features, params, var_scope='cnn')
        # return y_conv, None
        return build_residual_model(mode, inputs, \
            params, weak_learner_id)
    # default cnn
    y_conv, _ = lenet(features, params, var_scope='cnn')
    _, _ = lenet(features, params, var_scope='c_cnn')
    return y_conv, None

def model_fn(mode, inputs, params, reuse=False, weak_learner_id=0):
    """Model function defining the graph operations.
    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights
    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    is_test = (mode == 'test')
    weak_learner_id = int(weak_learner_id)
    # test will calculate NDCG and ERR directly
    # !!! (for real application please add constraints)
    labels = inputs['labels']
    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        # Compute the output distribution of the model and the predictions
        predictions, calcualted_loss = build_model(mode, inputs, params, \
                weak_learner_id=weak_learner_id)
        if not is_test:
            with tf.name_scope('loss'):
                # calcualted_loss = tf.Print(calcualted_loss, [calcualted_loss], message='calcualted_loss is \n')
                loss = get_loss(predictions, labels, params, calcualted_loss)
                if params.use_regularization:
                    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    loss += tf.reduce_sum(reg_losses)            
        if is_training:
            with tf.name_scope('adam_optimizer'):
                global_step = tf.train.get_or_create_global_step()
                optimizer = tf.train.AdamOptimizer(params.learning_rate)
                gradients, variables = zip(*optimizer.compute_gradients(loss))
                gradients, _ = tf.clip_by_global_norm(gradients, params.gradient_clip_value)
                train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
        
        with tf.name_scope('accuracy'):
            argmax_predictions = tf.argmax(predictions, 1)
            argmax_labels = tf.argmax(labels, 1)
            correct_prediction = tf.equal(argmax_predictions, argmax_labels)
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)

            # accuracy_per_class = tf.metrics.mean_per_class_accuracy(labels, predictions, \
            #     params.num_classes)
    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.mean(accuracy),
            # 'accuracy_pc': accuracy_per_class
        }
        tf.summary.scalar('accuracy', accuracy)
        if not is_test:
            # Summaries for training and validation
            metrics['loss'] = tf.metrics.mean(loss)
            # metrics['calculated_loss'] = tf.reduce_mean(inputs['calculated_loss'])
            tf.summary.scalar('loss', loss)
         
    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])
    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)
    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.local_variables_initializer(), \
        tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec["predictions"] = predictions
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()
    if is_training:
        model_spec['train_op'] = train_op
        model_spec['loss'] = loss
    return model_spec

def get_loss(predicted_scores, labels,
             params, calcualted_loss=None):
    """ 
    Return loss based on loss_function_str
    Note: this is for models that have real loss functions
    """
    def _cnn():
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                        logits=predicted_scores)
        loss = tf.reduce_mean(cross_entropy)
        return loss
    def _boost():
        return calcualted_loss
    def _retrain_regu():
        return _cnn() + calcualted_loss

    options = {
            'cnn': _cnn,
            'boost': _cnn,
            'retrain_regu': _retrain_regu
    }
    loss_function_str = params.loss_fn
    return options[loss_function_str]()
