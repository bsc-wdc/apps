# Python3 compatibility
from __future__ import division

import sys

from pycompss.api.parameter import *
from six.moves import range

from collections import Counter

import numpy as np
from math import sqrt
from pandas import read_csv
from pycompss.api.task import task


def get_feature_file(path, index):
    return path + 'x_' + str(index) + '.dat'


@task(returns=object)
def get_feature(path, i):
    print("@task get_feature")
    return read_csv(get_feature_file(path, i), header=None, squeeze=True)


@task(returns=np.ndarray)
def sample_selection(n_instances):
    print("@task sample_selection")
    bootstrap = np.random.choice(n_instances, size=n_instances, replace=True)
    bootstrap.sort()
    return bootstrap


def feature_selection(n_features):
    return np.random.choice(n_features, size=int(sqrt(n_features)), replace=False)


@task(returns=object)
def get_y(path):
    print("@task get_y")
    return read_csv(path + 'y.dat', header=None, squeeze=True)


def gini_index(counter, size):
    return 1 - sum((counter[key]/size)**2 for key in counter)


# Maximizing the Gini gain is equivalent to minimizing this weighted_sum
def gini_weighted_sum(l_frequencies, l_size, r_frequencies, r_size):
    weighted_sum = 0
    if l_size:
        weighted_sum += l_size*gini_index(l_frequencies, l_size)
    if r_size:
        weighted_sum += r_size*gini_index(r_frequencies, r_size)
    return weighted_sum


@task(returns=list)
def test_splits(sample, feature, y):
    print("@task test_splits")
    sort_indices = feature[sample].argsort()
    l_frequencies = Counter()
    l_size = 0
    r_frequencies = Counter(y[sample])
    r_size = len(sample)
    min_score = sys.float_info.max
    b_value = None
    for i in sort_indices:
        s = sample[i]
        l_frequencies[y[s]] += 1
        r_frequencies[y[s]] -= 1
        l_size += 1
        r_size -= 1
        try:
            s_next = sample[i+1]
            if feature[s] == feature[s_next]:
                continue
        except IndexError:  # Last element
            pass

        score = gini_weighted_sum(l_frequencies, l_size, r_frequencies, r_size)
        print(score)
        if score < min_score:
            min_score = score
            try:
                b_value = (feature[s] + feature[sample[i + 1]])/2
            except IndexError:  # Last element
                b_value = np.float64(np.inf)
    return tuple([min_score, b_value])


@task(priority=True, returns=(list, list))
def get_groups(sample, path, index, value):
    print("@task get_groups")
    left = []
    right = []
    if len(sample) > 0:
        feature = read_csv(get_feature_file(path, index), header=None, squeeze=True)
        for i in sample:
            if feature[i] < value:
                left.append(i)
            else:
                right.append(i)
    return left, right


class Node(object):

    def __init__(self, tree_path, index, value):
        self.tree_path = tree_path
        self.index = index
        self.value = value

    def to_json(self):
        return ('{{"type": "NODE", "tree_path": "{}", '
                '"index": {}, "value": {}}}'
                .format(self.tree_path, self.index, self.value))


@task(priority=True, returns=(Node, int, float))
def get_best_split(tree_path, features, *scores_and_values):
    print("@task get_best_split")
    min_score = sys.float_info.max
    b_index = None
    b_value = None
    for i in range(len(scores_and_values)):
        score, value = scores_and_values[i]
        if score < min_score:
            min_score = score
            b_index = i
            b_value = value
    if b_index is None:
        b_feature = None
    else:
        b_feature = features[b_index]
    if b_value is not None:
        b_value = b_value.item()
    return Node(tree_path, b_feature, b_value), b_feature, b_value


class Leaf(object):

    def __init__(self, tree_path, size, frequencies, mode):
        self.tree_path = tree_path
        self.size = size
        self.frequencies = frequencies
        self.mode = mode

    def to_json(self):
        frequencies_str = ', '.join(map('%r: %r'.__mod__, self.frequencies.most_common()))
        frequencies_str = '{' + frequencies_str + '}'
        return ('{{"type": "LEAF", "tree_path": "{}", '
                '"size": {}, "mode": {}, "frequencies": {}}}'
                .format(self.tree_path, self.size, self.mode, frequencies_str))


@task(returns=Leaf)
def build_leaf(sample, y, tree_path):
    print('@task build_leaf')
    frequencies = Counter(y[sample])
    most_common = frequencies.most_common(1)
    if most_common:
        mode = most_common[0][0]
    else:
        mode = None
    return Leaf(tree_path, len(sample), frequencies, mode)


def compute_split(tree_path, sample, features, path, y):
    n_features = len(features)
    index_selection = feature_selection(n_features)
    scores_and_values = []
    for i in index_selection:
        scores_and_values.append(test_splits(sample, features[i], y))
    node, index, value = get_best_split(tree_path, index_selection, *scores_and_values)
    left_group, right_group = get_groups(sample, path, index, value)
    return node, left_group, right_group


@task(file_out=FILE_INOUT)
def flush_nodes_task(file_out, *nodes_to_persist):
    print('@task flush_nodes_task')
    with open(file_out, "a") as tree_file:
        for node in nodes_to_persist:
            tree_file.write(node.to_json() + '\n')


def flush_nodes(file_out, nodes_to_persist):
    flush_nodes_task(file_out, *nodes_to_persist)
    del nodes_to_persist[:]


class DecisionTree:

    def __init__(self, path_in, n_instances, n_features):
        """
        Decision tree with distributed splits using pyCOMPSs.

        :param path_in: Path of the dataset directory.
        :param n_instances: Number of instances in the sample.
        :param n_features: Number of attributes in the sample.
        """
        self.path_in = path_in
        self.n_instances = n_instances
        self.n_features = n_features

    def fit(self, max_depth, path_out, name):
        """
        Fits the DecisionTree.

        :param max_depth: Depth of the decision tree.
        :param path_out: Path of the output directory.
        :param name: Name of the output file.
        """
        tree_sample = sample_selection(self.n_instances)
        features = []  # Chunking would require task refactoring
        for i in range(self.n_features):
            features.append(get_feature(self.path_in, i))
        y = get_y(self.path_in)
        nodes_to_split = [('/', tree_sample, 1)]
        file_out = path_out + name
        open(file_out, 'w').close()  # Create new empty file deleting previous content
        nodes_to_persist = []
        while nodes_to_split:
            tree_path, sample, depth = nodes_to_split.pop()
            node, left_group, right_group = compute_split(tree_path, sample, features, self.path_in, y)
            nodes_to_persist.append(node)
            if depth < max_depth:
                nodes_to_split.append((tree_path + 'R', right_group, depth + 1))
                nodes_to_split.append((tree_path + 'L', left_group, depth + 1))
            else:
                left = build_leaf(left_group, y, tree_path + 'L')
                nodes_to_persist.append(left)

                right = build_leaf(right_group, y, tree_path + 'R')
                nodes_to_persist.append(right)
            if len(nodes_to_persist) >= 10000:
                flush_nodes(file_out, nodes_to_persist)
        flush_nodes(file_out, nodes_to_persist)


def main():
    tree = DecisionTree('/home/bscuser/datasets/dt_test/', 4, 2)
    #
    # import cProfile
    # pr = cProfile.Profile()
    # pr.disable()
    # pr.enable()
    #
    tree.fit(2, '/home/bscuser/random_forest/', 'tree_0')
    # compss_barrier()
    #
    # pr.disable()
    # # pr.print_stats(sort='time')
    #
    # # tree.get_and_print()
