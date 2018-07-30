from __future__ import division
import argparse
import os
import time
import numpy as np

from pycompss.api.api import compss_barrier, compss_open
from pycompss.api.parameter import DIRECTION
from pycompss.runtime.binding import get_file

from forest import RandomForestClassifier
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier as SklearnRFClassifier
from sklearn.tree import export_graphviz


def main():
    initial_time = time.time()
    parser = argparse.ArgumentParser(description='Random forest classifier.')

    # RandomForestClassifier params
    parser.add_argument('--path_in', help='Path of the dataset directory.')
    parser.add_argument('--n_instances', type=int, help='Number of instances in the sample.')
    parser.add_argument('--n_features', type=int, help='Number of attributes in the sample.')
    parser.add_argument('--path_out', help='Path of the output directory.')
    parser.add_argument('--n_estimators', type=int, help='Number of trees to build.')
    parser.add_argument('--max_depth', type=int, default=None, help='Depth of the decision tree.')
    parser.add_argument('--distr_depth', type=int, default=None, help='Tasks are distributed up to this depth.')

    args = parser.parse_args()

    forest = RandomForestClassifier(args.path_in, args.n_instances, args.n_features,
                                    args.path_out, args.n_estimators, args.max_depth, args.distr_depth)
    forest.fit()

    compss_barrier()

    fit_time = time.time()

    print(args)
    print('Fit time: ' + str(fit_time - initial_time))

    for tree in forest.trees:
        tree_file = get_file(tree.path_out + tree.name_out, DIRECTION.IN)
        tree.path_out, tree.name_out = os.path.split(tree_file)
        if tree.path_out[-1] != os.sep:
            tree.path_out = tree.path_out + os.sep

    get_files_time = time.time()
    print('Open time: ' + str(get_files_time - fit_time))

    predict = False
    try:
        x_test = np.load(args.path_in + 'x_test.npy', allow_pickle=False)
        y_real = read_csv(args.path_in + 'y_test.dat', header=None, dtype=object, squeeze=True)
        predict = True
    except (IOError, ValueError):
        print 'No files for prediction found'
    if predict:
        predict_start_time = time.time()
        y_predicted = forest.predict()
        predict_time = time.time()
        accuracy = (np.count_nonzero(y_predicted == y_real)) / len(y_real)

        print('Predict time: ' + str(predict_time - predict_start_time))
        print 'Accuracy: ', accuracy

    # Compare to sklearn
    # sk_start_time = time.time()
    # sk_forest = SklearnRFClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
    #                                 max_features=args.n_features)
    # x_train = np.load(args.path_in + 'x.npy', allow_pickle=False)
    # y_train = read_csv(args.path_in + 'y.dat', header=None, dtype=object, squeeze=True)
    # sk_forest.fit(x_train, y_train)
    # sk_fit_time = time.time()
    # print 'SK_fit_time: ', str(sk_fit_time - sk_start_time)
    # if predict:
    #     y_predicted = sk_forest.predict(x_test)
    #     sk_accuracy = (np.count_nonzero(y_predicted == y_real)) / len(y_real)
    #     print 'SK_accuracy: ', sk_accuracy
    # for i, tree in enumerate(sk_forest.estimators_):
    #     export_graphviz(tree, open('sk_tree_' + str(i), 'w'), precision=6)


if __name__ == "__main__":
    main()
