# Python3 compatibility imports
from __future__ import division

from itertools import izip, izip_longest, tee
from sys import float_info

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


def get_features_file(path):
    return path + 'x_t.npy'


def get_feature_path(path, index):
    return path + 'x_' + str(index) + '.dat'


@task(returns=object)
def get_feature_task(*args):
    print("@task get_feature_task")
    return get_feature(*args)


def get_feature(path, i):
    return read_csv(get_feature_path(path, i), header=None, squeeze=True).values


@task(returns=np.ndarray)
def sample_selection(n_instances):
    print("@task sample_selection")
    bootstrap = np.random.choice(n_instances, size=n_instances, replace=True)
    bootstrap.sort()
    return bootstrap


def feature_selection(n_features):
    return np.random.choice(n_features, size=m_try(n_features), replace=False)


def m_try(n_features):
    return int(sqrt(n_features))


@task(returns=object)
def get_y(path):
    print("@task get_y")
    return read_csv(path + 'y.dat', dtype=object, header=None, squeeze=True).values


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


def test_split(sample, y_s, feature):
    min_score = float_info.max
    b_value = None
    l_frequencies = Counter()
    l_size = 0
    r_frequencies = Counter(y_s)
    r_size = len(sample)
    data = sorted(izip(feature[sample], y_s), key=lambda e: e[0])
    i1, i2 = tee(data)
    next(i2, None)
    pairs_iter = izip_longest(i1, i2, fillvalue=None)
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
def test_splits(sample, y_s, feature_indices, *features):
    print("@task test_splits")
    min_score = float_info.max
    b_value = None
    b_index = None
    for t in range(len(feature_indices)):
        feature = features[t]
        score, value = test_split(sample, y_s, feature)
        if score < min_score:
            min_score = score
            b_index = feature_indices[t]
            b_value = value
    return min_score, b_value, b_index


@task(features_file=FILE_IN, returns=(Node, list, list, list, list))
def get_best_split(tree_path, sample, y_s, features_file, *scores_and_values_and_indices):
    print("@task get_best_split")
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
    node = Node(tree_path, b_index, b_value)
    left_group, y_l, right_group, y_r = get_groups(sample, y_s, features_file, b_index, b_value)
    return node, left_group, y_l, right_group, y_r


def get_groups(sample, y_s, features_file, index, value):
    if index is None:
        return sample, y_s, np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    features_mmap = np.load(features_file, mmap_mode='r', allow_pickle=False)
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


def compute_split(tree_path, sample, depth, n_instances, features, path, y_s):
    n_features = len(features)
    index_selection = feature_selection(n_features)
    chunk = max(1, int(floor(10000 * 2 ** (depth - 1) / n_instances)))
    scores_and_values_and_indices = []
    while (len(index_selection)) > 0:
        indices_to_test = index_selection[:chunk]
        index_selection = index_selection[chunk:]
        scores_and_values_and_indices.append(
            test_splits(sample, y_s, indices_to_test, *[features[i] for i in indices_to_test]))
    features_file = get_features_file(path)
    node, left_group, y_l, right_group, y_r = get_best_split(tree_path, sample, y_s, features_file,
                                                             *scores_and_values_and_indices)
    return node, left_group, y_l, right_group, y_r


def compute_split_simple(tree_path, sample, n_features, features_file, y_s):
    index_selection = feature_selection(n_features)
    b_score = float_info.max
    b_index = None
    b_value = None
    for index in index_selection:
        features_mmap = np.load(features_file, mmap_mode='r', allow_pickle=False)
        feature = features_mmap[index]
        score, value = test_split(sample, y_s, feature)
        if score < b_score:
            b_score, b_value, b_index = score, value, index
    node = Node(tree_path, b_index, b_value)
    left_group, y_l, right_group, y_r = get_groups(sample, y_s, features_file, b_index, b_value)
    return node, left_group, y_l, right_group, y_r


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


@task(features_file=FILE_IN, returns=list)
def build_subtree(sample, y_s, tree_path, max_depth, n_features, features_file):
    print("@task build_subtree")
    if not sample.size:
        return []
    nodes_to_split = [(tree_path, sample, y_s, 1)]
    node_list_to_persist = []
    while nodes_to_split:
        tree_path, sample, y_s, depth = nodes_to_split.pop()
        node, left_group, y_l, right_group, y_r = compute_split_simple(tree_path, sample, n_features, features_file, y_s)
        if not left_group.size or not right_group.size:
            leaf = build_leaf(y_s, tree_path)
            node_list_to_persist.append(leaf)
        else:
            node_list_to_persist.append(node)
            if depth < max_depth:
                nodes_to_split.append((tree_path + 'R', right_group, y_r, depth + 1))
                nodes_to_split.append((tree_path + 'L', left_group, y_l, depth + 1))
            else:
                left = build_leaf(y_l, tree_path + 'L')
                node_list_to_persist.append(left)

                right = build_leaf(y_r, tree_path + 'R')
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
        features_file = get_features_file(self.path_in)
        for i in range(self.n_features):
            features.append(get_feature_task(self.path_in, i))
        y = get_y(self.path_in)
        nodes_to_split = [('/', tree_sample, y, 1)]
        file_out = self.path_out + self.name_out
        open(file_out, 'w').close()  # Create new empty file deleting previous content
        nodes_to_persist = []
        while nodes_to_split:
            tree_path, sample, y_s, depth = nodes_to_split.pop()
            node, left_group, y_l, right_group, y_r = compute_split(tree_path, sample, depth, self.n_instances,
                                                                    features, self.path_in, y_s)
            nodes_to_persist.append(node)
            if depth < self.max_depth // 2:
                nodes_to_split.append((tree_path + 'R', right_group, y_r, depth + 1))
                nodes_to_split.append((tree_path + 'L', left_group, y_l, depth + 1))
            else:
                left_subtree_nodes = build_subtree(left_group, y_l, tree_path + 'L', self.max_depth - depth,
                                                   len(features), features_file)
                nodes_to_persist.append(left_subtree_nodes)

                right_subtree_nodes = build_subtree(right_group, y_r, tree_path + 'R', self.max_depth - depth,
                                                    len(features), features_file)
                nodes_to_persist.append(right_subtree_nodes)
            if len(nodes_to_persist) >= 1000:
                flush_nodes(file_out, nodes_to_persist)
        flush_nodes(file_out, nodes_to_persist)

    def predict(self):
        pass

    def predict_probabilities(self):
        pass
