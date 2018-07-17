from __future__ import division
import os
import json
import numpy as np


def predict(file_name, test_data):
    if len(test_data.shape) == 1:
        leaf = get_leaf(file_name, test_data)
        return leaf['mode']
    elif len(test_data.shape) == 2:
        n_samples = test_data.shape[0]
        res = np.zeros((n_samples, 1))
        for i, test_instance in enumerate(test_data):
            leaf = get_leaf(file_name, test_instance)
            res[i] = leaf['mode']
        return res
    else:
        raise ValueError


def predict_probabilities(file_name, test_data, n_classes):
    if len(test_data.shape) == 1:
        leaf = get_leaf(file_name, test_data)
        res = np.zeros((n_classes,))
        for cla, freq in leaf['frequencies'].iteritems():
            res[int(cla)] = freq
        return res/leaf['size']
    elif len(test_data.shape) == 2:
        n_samples = test_data.shape[0]
        res = np.zeros((n_samples, n_classes))
        for i, test_instance in enumerate(test_data):
            leaf = get_leaf(file_name, test_instance)
            for cla, freq in leaf['frequencies'].iteritems():
                res[i, int(cla)] = freq
            res[i] /= leaf['size']
        return res
    else:
        raise ValueError


def get_leaf(file_name, instance):
    tree_path = '/'
    node, pos = get_node(file_name, tree_path, 0)
    while True:
        node_type = node['type']
        if node_type == 'LEAF':
            return node
        elif node_type == 'NODE':
            if instance[node['index']] > node['value']:
                tree_path += 'R'
            else:
                tree_path += 'L'
            node, pos = get_node(file_name, tree_path, pos)
        else:
            raise Exception('Invalid node')


def get_node(file_name, tree_path, start):
    f_p = find(file_name, tree_path, start=start)
    if f_p == -1:
        raise Exception('Node not found')
    with open(file_name, 'rb') as f:
        f.seek(f_p-15)
        line = f.readline()
        node = json.loads(line)
        return node, f.tell()


def find(file_name, s, start=0):
    with open(file_name, 'rb') as f:
        file_size = os.path.getsize(file_name)
        bsize = 4096
        if start > 0:
            f.seek(start)
        overlap = len(s) - 1
        while True:
            if overlap <= f.tell() < file_size:
                f.seek(f.tell() - overlap)
            buffer_in = f.read(bsize)
            if buffer_in:
                pos = buffer_in.find(s)
                if pos >= 0:
                    return f.tell() - (len(buffer_in) - pos)
            else:
                return -1


if __name__ == '__main__':
    tree_file = '/home/bscuser/git/apps/python/pycompss_lib/ml/classification/random_forest/full/sample/tree_0'
    node1, pointer1 = get_node(tree_file, '/L', 0)
    node2, pointer2 = get_node(tree_file, '/LL', pointer1)
    print node1, node2
    print predict_probabilities(tree_file, np.arange(25) / 16, 3)
    my_data = np.load('/home/bscuser/datasets/dt_test_4/x.npy', allow_pickle=False)
    a= predict_probabilities(tree_file, my_data, 3)
