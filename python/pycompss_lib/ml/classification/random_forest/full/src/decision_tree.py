# Python3 compatibility imports
from __future__ import division
from sys import float_info

from pycompss.api.parameter import *
from six.moves import range

from collections import Counter
from numba import jit

import numpy as np
from numpy.lib import format
from math import sqrt, frexp
from pandas import read_csv
from pandas.api.types import CategoricalDtype
from pycompss.api.task import task
from pycompss.api.api import compss_delete_object

import prediction
from test_split import test_split


class InternalNode(object):

    def __init__(self, tree_path=None, index=None, value=None):
        self.tree_path = tree_path
        self.index = index
        self.value = value

    def to_json(self):
        return ('{{"tree_path": "{}", "type": "NODE", '
                '"index": {}, "value": {}}}'
                .format(self.tree_path, self.index, self.value))


class Leaf(object):

    def __init__(self, tree_path=None, size=None, frequencies=None, mode=None):
        self.tree_path = tree_path
        self.size = size
        self.frequencies = frequencies
        self.mode = mode

    def to_json(self):
        frequencies_str = ', '.join(map('"%r": %r'.__mod__, self.frequencies.items()))
        frequencies_str = '{' + frequencies_str + '}'
        return ('{{"tree_path": "{}", "type": "LEAF", '
                '"size": {}, "mode": {}, "frequencies": {}}}'
                .format(self.tree_path, self.size, self.mode, frequencies_str))


def get_features_file(path):
    return path + 'x_t.npy'


@task(features_file=FILE_IN, returns=object)
def get_feature_task(features_file, i):
    with open(features_file, mode='rb') as fp:
        version = format.read_magic(fp)
        shape, fortran_order, dtype = format._read_array_header(fp, version)
        n_samples = shape[1]
        fp.seek(i*n_samples*dtype.itemsize, 1)
        return np.fromfile(fp, dtype=dtype, count=n_samples)


def get_feature_mmap(features_file, i):
    return get_features_mmap(features_file)[i]


def get_features_mmap(features_file):
    return np.load(features_file, mmap_mode='r', allow_pickle=False)


@task(priority=True, returns=2)
def sample_selection(n_instances, y_codes, bootstrap):
    if bootstrap:
        selection = np.random.choice(n_instances, size=n_instances, replace=True)
        selection.sort()
        return selection, y_codes[selection]
    else:
        return np.arange(n_instances), y_codes


def feature_selection(feature_indices, m_try):
    return np.random.choice(feature_indices, size=min(m_try, len(feature_indices)), replace=False)


@task(returns=3)
def get_y(path):
    y = read_csv(path + 'y.dat', dtype=CategoricalDtype(), header=None, squeeze=True).values
    return y, y.codes, len(y.categories)


@task(returns=tuple)
def test_splits(sample, y_s, n_classes, feature_indices, *features):
    min_score = float_info.max
    b_value = None
    b_index = None
    for t in range(len(feature_indices)):
        feature = features[t]
        score, value = test_split(sample, y_s, feature, n_classes)
        if score < min_score:
            min_score = score
            b_index = feature_indices[t]
            b_value = value
    return min_score, b_value, b_index


@task(features_file=FILE_IN, returns=(InternalNode, list, list, list, list))
def get_best_split(tree_path, sample, y_s, features_file, *scores_and_values_and_indices):
    min_score = float_info.max
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
    node = InternalNode(tree_path, b_index, b_value)
    features_mmap = np.load(features_file, mmap_mode='r', allow_pickle=False)
    left_group, y_l, right_group, y_r = get_groups(sample, y_s, features_mmap, b_index, b_value)
    return node, left_group, y_l, right_group, y_r


def get_groups(sample, y_s, features_mmap, index, value):
    if index is None:
        return sample, y_s, np.array([], dtype=np.int64), np.array([], dtype=np.int8)
    feature = features_mmap[index][sample]
    mask = feature < value
    left = sample[mask]
    right = sample[~mask]
    y_l = y_s[mask]
    y_r = y_s[~mask]
    return left, y_l, right, y_r


def build_leaf(y_s, tree_path):
    frequencies = Counter(y_s)
    most_common = frequencies.most_common(1)
    if most_common:
        mode = most_common[0][0]
    else:
        mode = None
    return Leaf(tree_path, len(y_s), frequencies, mode)


def compute_split_chunked(tree_path, sample, depth, features, features_file, y_s, n_classes, m_try):
    n_features = len(features)
    index_selection = feature_selection(range(n_features), m_try)
    chunk = 2**(min(depth, 20) - 1)
    scores_and_values_and_indices = []
    while (len(index_selection)) > 0:
        indices_to_test = index_selection[:chunk]
        index_selection = index_selection[chunk:]
        scores_and_values_and_indices.append(
            test_splits(sample, y_s, n_classes, indices_to_test, *[features[i] for i in indices_to_test]))
    node, left_group, y_l, right_group, y_r = get_best_split(tree_path, sample, y_s, features_file,
                                                             *scores_and_values_and_indices)
    return node, left_group, y_l, right_group, y_r


def compute_split(tree_path, sample, n_features, features_mmap, y_s, n_classes, m_try):
    split_ended = False
    tried_indices = []
    while not split_ended:
        index_selection = feature_selection(np.setdiff1d(np.arange(n_features), tried_indices), m_try)
        b_score = float_info.max
        b_index = None
        b_value = None
        for index in index_selection:
            feature = features_mmap[index]
            score, value = test_split(sample, y_s, feature, n_classes)
            if score < b_score:
                b_score, b_value, b_index = score, value, index
        left_group, y_l, right_group, y_r = get_groups(sample, y_s, features_mmap, b_index, b_value)
        if left_group.size and right_group.size:
            split_ended = True
            node = InternalNode(tree_path, b_index, b_value)
        else:
            tried_indices.extend(list(index_selection))
            if len(tried_indices) == n_features:
                split_ended = True
                node = build_leaf(y_s, tree_path)

    return node, left_group, y_l, right_group, y_r


def flush_nodes(file_out, nodes_to_persist):
    flush_nodes_task(file_out, *nodes_to_persist)
    for obj in nodes_to_persist:
        compss_delete_object(obj)
    del nodes_to_persist[:]


@task(file_out=FILE_INOUT)
def flush_nodes_task(file_out, *nodes_to_persist):
    with open(file_out, "a") as tree_file:
        for item in nodes_to_persist:
            if isinstance(item, (Leaf, InternalNode)):
                tree_file.write(item.to_json() + '\n')
            else:
                for node in item:
                    tree_file.write(node.to_json() + '\n')


@jit(nopython=True)
def is_constant(array):
    first = array[0]
    for a in array:
        if a != first:
            return False
    return True


@task(features_file=FILE_IN, returns=list)
def build_subtree(sample, y_s, n_features, tree_path, max_depth, n_classes, features_file, m_try):
    if not sample.size:
        return []
    features_mmap = np.load(features_file, mmap_mode='r', allow_pickle=False)
    nodes_to_split = [(tree_path, sample, y_s, 1)]
    node_list_to_persist = []
    while nodes_to_split:
        tree_path, sample, y_s, depth = nodes_to_split.pop()
        if is_constant(y_s):
            leaf = build_leaf(y_s, tree_path)
            node_list_to_persist.append(leaf)
        else:
            node, left_group, y_l, right_group, y_r = compute_split(tree_path, sample, n_features, features_mmap,
                                                                    y_s, n_classes, m_try)
            node_list_to_persist.append(node)
            if isinstance(node, InternalNode):
                if depth < max_depth:
                    nodes_to_split.append((tree_path + 'R', right_group, y_r, depth + 1))
                    nodes_to_split.append((tree_path + 'L', left_group, y_l, depth + 1))
                else:
                    left = build_leaf(y_l, tree_path + 'L')
                    node_list_to_persist.append(left)

                    right = build_leaf(y_r, tree_path + 'R')
                    node_list_to_persist.append(right)
    return node_list_to_persist


class DecisionTreeClassifier:

    def __init__(self, path_in, n_instances, n_features, path_out, name_out, max_depth=None, distr_depth=None,
                 bootstrap=False):
        """
        Decision tree with distributed splits using pyCOMPSs.

        :param path_in: Path of the dataset directory.
        :param n_instances: Number of instances in the sample.
        :param n_features: Number of attributes in the sample.
        :param path_out: Path of the output directory.
        :param name_out: Name of the output file.
        :param max_depth: Depth of the decision tree.
        :param distr_depth: Nodes are split in a distributed way up to this depth.
        :param bootstrap: Randomly select n_instances samples with repetition (used in random forests).
        """
        self.path_in = path_in
        self.n_instances = n_instances
        self.n_features = n_features
        self.path_out = path_out
        self.name_out = name_out
        self.max_depth = max_depth if max_depth is not None else np.inf
        self.distr_depth = distr_depth if distr_depth is not None else (frexp(self.n_instances)[1] - 1) // 3
        self.features = []
        self.y = None
        self.y_codes = None
        self.n_classes = None
        self.m_try = n_features
        self.bootstrap = bootstrap

    def fit(self):
        """
        Fits the DecisionTreeClassifier.
        """
        if self.y_codes is None:
            self.y, self.y_codes, self.n_classes = get_y(self.path_in)
        tree_sample, y_s = sample_selection(self.n_instances, self.y_codes, self.bootstrap)
        features_file = get_features_file(self.path_in)
        if not self.features:
            for i in range(self.n_features):
                self.features.append(get_feature_task(features_file, i))
        nodes_to_split = [('/', tree_sample, y_s, 1)]
        file_out = self.path_out + self.name_out
        open(file_out, 'w').close()  # Create new empty file deleting previous content
        nodes_to_persist = []
        while nodes_to_split:
            tree_path, sample, y_s, depth = nodes_to_split.pop()
            node, left_group, y_l, right_group, y_r = compute_split_chunked(tree_path, sample, depth, self.features,
                                                                            features_file, y_s, self.n_classes,
                                                                            self.m_try)
            compss_delete_object(sample)
            compss_delete_object(y_s)
            nodes_to_persist.append(node)
            if depth < self.distr_depth:
                nodes_to_split.append((tree_path + 'R', right_group, y_r, depth + 1))
                nodes_to_split.append((tree_path + 'L', left_group, y_l, depth + 1))
            else:
                left_subtree_nodes = build_subtree(left_group, y_l, self.n_features, tree_path + 'L',
                                                   self.max_depth - depth, self.n_classes, features_file, self.m_try)
                nodes_to_persist.append(left_subtree_nodes)
                compss_delete_object(left_group)
                compss_delete_object(y_l)

                right_subtree_nodes = build_subtree(right_group, y_r, self.n_features, tree_path + 'R',
                                                    self.max_depth - depth, self.n_classes, features_file, self.m_try)
                nodes_to_persist.append(right_subtree_nodes)
                compss_delete_object(right_group)
                compss_delete_object(y_r)

            if len(nodes_to_persist) >= 1000:
                flush_nodes(file_out, nodes_to_persist)
        
        flush_nodes(file_out, nodes_to_persist)

    def predict(self, x_test):
        """ Predicts class codes for the input data using a fitted tree and returns an integer or an array. """
        file_tree = self.path_out + self.name_out
        return prediction.predict(file_tree, x_test)

    def predict_probabilities(self, x_test):
        """ Predicts class probabilities by class code using a fitted tree and returns a 1D or 2D array. """
        file_tree = self.path_out + self.name_out
        return prediction.predict_probabilities(file_tree, x_test, self.n_classes)
