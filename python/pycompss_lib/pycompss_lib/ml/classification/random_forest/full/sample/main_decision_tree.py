from __future__ import division
import argparse
import os
import time

import numpy as np
from pandas import read_csv

from pycompss.api.api import compss_barrier, compss_open, compss_wait_on
from pycompss.api.parameter import DIRECTION
from pycompss.runtime.binding import get_file

from decision_tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier as SklearnDTClassifier


def main():
    initial_time = time.time()
    parser = argparse.ArgumentParser(description='Decision tree classifier.')

    # DecisionTree params
    parser.add_argument('--path_in', help='Path of the dataset directory.',)
    parser.add_argument('--n_instances', type=int, help='Number of instances in the sample.')
    parser.add_argument('--n_features', type=int, help='Number of attributes in the sample.')
    parser.add_argument('--path_out', help='Path of the output directory.')
    parser.add_argument('--name_out', help='Name of the output file.')
    parser.add_argument('--max_depth', type=int, help='Depth of the decision tree.')
    parser.add_argument('--distr_depth', type=int, default=None, help='Tasks are distributed up to this depth.')
    parser.add_argument('--try_features', default=None, help='Number of features to try at each split.')

    args = parser.parse_args()

    tree = DecisionTreeClassifier(args.path_in, args.n_instances, args.n_features, args.path_out, args.name_out,
                                  args.max_depth, args.distr_depth, False, args.try_features)
    tree.fit()

    compss_barrier()

    fit_time = time.time()

    print(args)
    print('Fit time: ' + str(fit_time - initial_time))

    tree_file = get_file(args.path_out + args.name_out, DIRECTION.IN)
    tree.path_out, tree.name_out = os.path.split(tree_file)
    if tree.path_out[-1] != os.sep:
        tree.path_out = tree.path_out + os.sep

    get_file_time = time.time()
    print('get_file time: ' + str(get_file_time - fit_time))

    tree.y = compss_wait_on(tree.y)

    predict = False
    try:
        x_test = np.load(args.path_in + 'x_test.npy', allow_pickle=False)
        y_real = read_csv(args.path_in + 'y_test.dat', header=None, dtype=object, squeeze=True)
        predict = True
    except (IOError, ValueError):
        print 'No files for prediction found'
    if predict:
        y_predicted = tree.y.categories[tree.predict(x_test)]
        predict_time = time.time()
        accuracy = (np.count_nonzero(y_predicted == y_real)) / len(y_real)
        print('Predict time: ' + str(predict_time - get_file_time))
        print 'Accuracy: ', accuracy

    # Compare to sklearn
    # sk_start_time = time.time()
    # sk_tree = SklearnDTClassifier()
    # x_train = np.load(args.path_in + 'x.npy', allow_pickle=False)
    # y_train = read_csv(args.path_in + 'y.dat', header=None, dtype=object, squeeze=True)
    # sk_tree.fit(x_train, y_train)
    # sk_fit_time = time.time()
    # print 'SK_time: ', str(sk_fit_time - sk_start_time)
    # if predict:
    #     y_predicted = sk_tree.predict(x_test)
    #     sk_accuracy = (np.count_nonzero(y_predicted == y_real)) / len(y_real)
    #     print 'SK_accuracy: ', sk_accuracy


if __name__ == "__main__":
    main()
