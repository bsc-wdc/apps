# Python3 compatibility imports
from __future__ import division

import itertools
import sys

from pycompss.api.parameter import *
from six.moves import range

from collections import Counter

import numpy as np
from math import sqrt, floor
from pandas import read_csv
from pycompss.api.task import task


class Node(object):

    def __init__(self, tree_path=None, index=None, value=None):
        self.tree_path = tree_path
        self.index = index
        self.value = value

    def to_json(self):
        return ('{{"type": "NODE", "tree_path": "{}", '
                '"index": {}, "value": {}}}'
                .format(self.tree_path, self.index, self.value))


class Leaf(object):

    def __init__(self, tree_path=None, size=None, frequencies=None, mode=None):
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
    return read_csv(path + 'y.dat', dtype=object, header=None, squeeze=True)


def gini_index(counter, size):
    return 1 - sum((counter[key] / size) ** 2 for key in counter)


# Maximizing the Gini gain is equivalent to minimizing this weighted_sum
def gini_weighted_sum(l_frequencies, l_size, r_frequencies, r_size):
    weighted_sum = 0
    if l_size:
        weighted_sum += l_size * gini_index(l_frequencies, l_size)
    if r_size:
        weighted_sum += r_size * gini_index(r_frequencies, r_size)
    return weighted_sum


def test_split(sample, y, feature):
    y_sample = y[sample]
    min_score = sys.float_info.max
    b_value = None
    l_frequencies = Counter()
    l_size = 0
    r_frequencies = Counter(y_sample)
    r_size = len(sample)
    data = sorted(zip(feature[sample], y_sample), key=lambda e: e[0])
    i1, i2 = itertools.tee(data)
    next(i2, None)
    pairs_iter = itertools.izip_longest(i1, i2, fillvalue=None)
    for el, next_el in pairs_iter:
        el_class = el[1]
        l_frequencies[el_class] += 1
        r_frequencies[el_class] -= 1
        l_size += 1
        r_size -= 1
        if next_el and el_class == next_el[1]:
            continue
        score = gini_weighted_sum(l_frequencies, l_size, r_frequencies, r_size)
        if score < min_score:
            min_score = score
            if next_el:
                b_value = (el[0] + next_el[0]) / 2
            else:  # Last element
                b_value = np.float64(np.inf)
    return min_score, b_value


@task(returns=tuple)
def test_splits(sample, y, feature_indices, *features):
    print("@task test_splits")
    min_score = sys.float_info.max
    b_value = None
    b_index = None
    for t in range(len(feature_indices)):
        feature = features[t]
        score, value = test_split(sample, y, feature)
        if score < min_score:
            min_score = score
            b_index = feature_indices[t]
            b_value = value
    return min_score, b_value, b_index


@task(priority=True, returns=(Node, int, float))
def get_best_split(tree_path, sample, path, *scores_and_values_and_indices):
    print("@task get_best_split")
    min_score = sys.float_info.max
    b_index = None
    b_value = None
    for i in range(len(scores_and_values_and_indices)):
        score, value, index = scores_and_values_and_indices[i]
        if score < min_score:
            min_score = score
            b_value = value
            b_index = index
    if b_value is not None:
        b_value = b_value.item()
    node = Node(tree_path, b_index, b_value)
    left_group, right_group = get_groups(sample, path, b_index, b_value)
    return node, left_group, right_group


def get_groups(sample, path, index, value):
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


def build_leaf(sample, y, tree_path):
    frequencies = Counter(y[sample])
    most_common = frequencies.most_common(1)
    if most_common:
        mode = most_common[0][0]
    else:
        mode = None
    return Leaf(tree_path, len(sample), frequencies, mode)


def compute_split(tree_path, sample, depth, n_instances, features, path, y):
    n_features = len(features)
    index_selection = feature_selection(n_features)
    chunk = max(1, int(floor(10000 * 2 ** (depth - 1) / n_instances)))
    scores_and_values_and_indices = []
    while (len(index_selection)) > 0:
        indices_to_test = index_selection[:chunk]
        index_selection = index_selection[chunk:]
        scores_and_values_and_indices.append(
            test_splits(sample, y, indices_to_test, *[features[i] for i in indices_to_test]))
    node, left_group, right_group = get_best_split(tree_path, sample, path, *scores_and_values_and_indices)
    return node, left_group, right_group


def compute_split_simple(tree_path, sample, n_features, path, y):
    index_selection = feature_selection(n_features)
    b_score = sys.float_info.max
    b_index = None
    b_value = None
    for index in index_selection:
        feature = read_csv(get_feature_file(path, index), header=None, squeeze=True)
        score, value = test_split(sample, y, feature)
        if score < b_score:
            b_score, b_value, b_index = score, value, index
    node = Node(tree_path, b_index, b_value)
    left_group, right_group = get_groups(sample, path, b_index, b_value)
    return node, left_group, right_group


def flush_nodes(file_out, nodes_to_persist):
    flush_nodes_task(file_out, *nodes_to_persist)
    del nodes_to_persist[:]


@task(file_out=FILE_INOUT)
def flush_nodes_task(file_out, *nodes_to_persist):
    print('@task flush_nodes_task')
    with open(file_out, "a") as tree_file:
        for item in nodes_to_persist:
            if isinstance(item, (Leaf, Node)):
                tree_file.write(item.to_json() + '\n')
            else:
                for node in item:
                    tree_file.write(node.to_json() + '\n')


@task(returns=list)
def build_subtree(sample, y, tree_path, max_depth, n_features, path_in):
    nodes_to_split = [(tree_path, sample, 1)]
    node_list_to_persist = []
    while nodes_to_split:
        tree_path, sample, depth = nodes_to_split.pop()
        node, left_group, right_group = compute_split_simple(tree_path, sample, n_features, path_in, y)
        if not left_group or not right_group:
            leaf = build_leaf(sample, y, tree_path)
            node_list_to_persist.append(leaf)
        else:
            node_list_to_persist.append(node)
            if depth < max_depth:
                nodes_to_split.append((tree_path + 'R', right_group, depth + 1))
                nodes_to_split.append((tree_path + 'L', left_group, depth + 1))
            else:
                left = build_leaf(left_group, y, tree_path + 'L')
                node_list_to_persist.append(left)

                right = build_leaf(right_group, y, tree_path + 'R')
                node_list_to_persist.append(right)
    return node_list_to_persist


class DecisionTree:

    def __init__(self, path_in, n_instances, n_features, path_out, name_out, max_depth=None):
        """
        Decision tree with distributed splits using pyCOMPSs.

        :param path_in: Path of the dataset directory.
        :param n_instances: Number of instances in the sample.
        :param n_features: Number of attributes in the sample.
        :param path_out: Path of the output directory.
        :param name_out: Name of the output file.
        :param max_depth: Depth of the decision tree.
        """
        self.path_in = path_in
        self.n_instances = n_instances
        self.n_features = n_features
        self.path_out = path_out
        self.name_out = name_out
        self.max_depth = max_depth

    def fit(self):
        """
        Fits the DecisionTree.
        """
        tree_sample = sample_selection(self.n_instances)
        features = []  # Chunking would require task refactoring
        for i in range(self.n_features):
            features.append(get_feature(self.path_in, i))
        y = get_y(self.path_in)
        nodes_to_split = [('/', tree_sample, 1)]
        file_out = self.path_out + self.name_out
        open(file_out, 'w').close()  # Create new empty file deleting previous content
        nodes_to_persist = []
        while nodes_to_split:
            tree_path, sample, depth = nodes_to_split.pop()
            node, left_group, right_group = compute_split(tree_path, sample, depth, self.n_instances, features,
                                                          self.path_in, y)
            nodes_to_persist.append(node)
            if depth < self.max_depth // 2:
                nodes_to_split.append((tree_path + 'R', right_group, depth + 1))
                nodes_to_split.append((tree_path + 'L', left_group, depth + 1))
            else:
                left_subtree_nodes = build_subtree(left_group, y, tree_path + 'L', self.max_depth - depth,
                                                   len(features), self.path_in)
                nodes_to_persist.append(left_subtree_nodes)

                right_subtree_nodes = build_subtree(right_group, y, tree_path + 'R', self.max_depth - depth,
                                                    len(features), self.path_in)
                nodes_to_persist.append(right_subtree_nodes)
            if len(nodes_to_persist) >= 1000:
                flush_nodes(file_out, nodes_to_persist)
        flush_nodes(file_out, nodes_to_persist)

    def predict(self):
        pass

    def predict_probabilities(self):
        pass
