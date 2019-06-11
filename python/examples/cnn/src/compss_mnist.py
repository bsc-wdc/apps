#!/usr/bin/python
#
#  Copyright 2002-2019 Barcelona Supercomputing Center (www.bsc.es)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
from itertools import product

import tensorflow as tf
from pycompss.api.task import task

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', "data/", "Directory to load/store data")
flags.DEFINE_string('checkpoint_dir', "models/",
                    "Directory to load/store model checkpoint")
flags.DEFINE_string('checkpoint_file', "models/advanced_mnist.ckpt",
                    "File to load/store model checkpoint")
flags.DEFINE_integer('batch_size', 50, "Number of elements used in each batch")
flags.DEFINE_integer('max_iterations', 200,
                     "Max iterations to be performed per model")
flags.DEFINE_integer('task_batch', 100,
                     "Number of iterations to be performed per task")


def memory_usage_psutil():
    # return the memory usage in MB
    import os
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


@task(returns=dict)
def do_training_step(model, iteration, task_iterations, batch_size,
                     train_images, train_labels):
    from time import time
    from utils import DataSet

    iteration *= task_iterations
    start = time()

    train = DataSet(train_images, train_labels)

    rate, factor = model['rate_factor']
    learning_rate = rate / factor
    neurons = model['neuron_number']

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # First Convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second Convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely Connected layer
    W_fc1 = weight_variable([7 * 7 * 64, neurons])
    b_fc1 = bias_variable([neurons])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout
    W_fc2 = weight_variable([neurons, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()  # defaults to saving all variables

    sess = tf.Session()

    if iteration == 0:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
    else:
        try:
            saver.restore(sess, model['path'])
            import os
            os.remove(model['path'])
            os.remove("%s.meta" % model['path'])
            print(
                "Model %s correctly loaded i:%s" % (model['path'], iteration))
        except Exception as e:
            print("No checkpoint for model %s found in iter %s\n%s" % (
            model['path'], iteration, e))
            sys.exit(1)

    if not model:
        model = dict()

    t1 = time()
    import numpy as np
    step_times = np.array([])
    for i in range(iteration, iteration + task_iterations):
        ls = time()
        batch_offset = i * batch_size
        batch = train.get_batch(batch_offset, batch_size)

        train_step.run(session=sess,
                       feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        # print("Iteration %s - %s s " % (i, time()-ls))
        step_times = np.append(step_times, (time() - ls))
    print("Loop\n Avg: %s\n Max: %s\n Min: %s" % (
    step_times.mean(), step_times.max(), step_times.min()))
    print("Loop time %s" % (time() - t1))
    save_path = saver.save(sess, "%s_%s" % (model['base_path'], iteration))

    model['path'] = save_path

    training_accuracy = accuracy.eval(session=sess,
                                      feed_dict={x: batch[0], y_: batch[1],
                                                 keep_prob: 1.0})
    model['train_accuracy'] = training_accuracy

    end = time()
    print(
        "Training stats:\n - Neurons's number: %s\n - Learning rate: %s\n - Model: %s\n - Time: %s\n" % \
        (neurons, learning_rate, save_path, (end - start)))

    return model


@task(returns=float)
def do_testing(model, test_images, test_labels):
    from time import time
    from utils import DataSet

    start = time()

    rate, factor = model['rate_factor']
    learning_rate = rate / factor
    neurons = model['neuron_number']
    test = DataSet(test_images, test_labels)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # First Convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second Convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely Connected layer
    W_fc1 = weight_variable([7 * 7 * 64, neurons])
    b_fc1 = bias_variable([neurons])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout
    W_fc2 = weight_variable([neurons, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()  # defaults to saving all variables

    sess = tf.Session()

    try:
        saver.restore(sess, model['path'])
        print("Model %s correctly loaded" % model['path'])
    except Exception as e:
        print("No checkpoint found\n%s" % e)
        sys.exit(1)

    test_accuracy = accuracy.eval(session=sess,
                                  feed_dict={x: test.images, y_: test.labels,
                                             keep_prob: 1.0})
    model['previous_test'] = model['test_accuracy']
    model['test_accuracy'] = test_accuracy

    end = time()
    print(
        "Testing stats:\n - Neurons's number: %s\n - Learning rate: %s\n - Test accuracy: %s\n - Model: %s\n - Time: %s\n" % \
        (neurons, learning_rate, test_accuracy, model['path'], (end - start)))

    return model


@task(returns=bool)
def continue_exploration(model):
    n = model['neuron_number']
    r, d = model['rate_factor']
    l = r / d
    t = model['train_accuracy']
    test = model['test_accuracy']
    prev = model['previous_test']
    print(
        "Model, neuron number: %d, learning rate: %f, train accuracy: %f, test accuracy: %f, previous test: %f" % (
        n, l, t, test, prev))
    if test >= prev:
        return True
    else:
        return False


def init_models(base_dir, parametrizations):
    models = []
    from uuid import uuid4
    id = uuid4()
    print("Initializing models with id: %s" % id)
    for n, l in parametrizations:
        path = "%s/%scompss_model_%s_%s_%s.ckpt" % (
        base_dir, FLAGS.checkpoint_dir, n, l, id)
        model = dict()
        model['path'] = path
        model['base_path'] = path
        model['neuron_number'] = n
        model['rate_factor'] = (l, 100000.0)
        model['train_accuracy'] = 0
        model['test_accuracy'] = 0
        model['previous_test'] = 0
        models.append(model)
        print(" - Model\n %s" % model)
    return models


@task()
def write_results(models, test_accuracies):
    name = "accuracies"
    v = 0

    while os.path.isfile('%s.%s' % (name, v)):
        v += 1

    with open('%s.%s' % (name, v), 'w') as f:
        f.write("Neurons, Rate")
        for i in range(len(models)):
            r, factor = models[i]['rate_factor']
            learning_rate = r / factor
            neurons = models[i]['neuron_number']
            f.write("\n%s, %s" % (neurons, learning_rate))
            print(" - Model [%s, %s]: %s" % (
            neurons, learning_rate, test_accuracies[i]))
            f.write(", %s" % test_accuracies[i])


def main(working_dir, number_of_models):
    from utils import read_sets
    from files import get_full_path
    from pycompss.api.api import compss_wait_on

    data_dir = get_full_path(FLAGS.data_dir)

    print("Data dir %s" % data_dir)
    train_data, test_data = read_sets(data_dir, one_hot=True)

    neurons = [1024, 1280, 1440, 1600]
    learning_rates = [1, 10, 25, 50, 75, 100, 125, 150]

    iter_test = 5

    task_iterations = FLAGS.task_batch

    parametrizations = list(product(neurons, learning_rates))
    parametrizations = parametrizations[:number_of_models]

    batch_size = FLAGS.batch_size
    max_training_iterations = (FLAGS.max_iterations)

    models = init_models(working_dir, parametrizations)

    print(
        "Parametrizations to be explored: %s with %s iterations and %s loops per task "
        % (parametrizations, max_training_iterations, task_iterations))
    n_tests = 1 + max_training_iterations / iter_test

    for iter in range(0, max_training_iterations):
        for (i, (neuron_number, learning_rate)) in enumerate(parametrizations):
            print("[%s] - Queueing %s - %s" % (
            iter, neuron_number, learning_rate,))
            if compss_wait_on(continue_exploration(models[i])):
                models[i] = do_training_step(models[i], iter, task_iterations,
                                             batch_size, train_data[0],
                                             train_data[1])
                if iter % iter_test == 0:
                    models[i] = do_testing(models[i], test_data[0],
                                           test_data[1])
            else:
                parametrizations.remove((neuron_number, learning_rate))

    test_accuracies = [0] * len(models)
    for i in range(len(models)):
        test_accuracies[i] = do_testing(models[i], test_data[0], test_data[1])

    write_results(models, test_accuracies)


if __name__ == "__main__":
    base_path = sys.argv[1]
    number_of_models = int(sys.argv[2])

    from time import time

    t_s = time()
    main(base_path, number_of_models)
    t_f = time()
    print(" - Execution time: %s" % (t_f - t_s))
