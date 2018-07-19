from __future__ import division
import argparse
import time
import numpy as np

from pycompss.api.api import compss_barrier, compss_open
from forest import RandomForestClassifier
from pandas import read_csv


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

    for i in range(args.n_estimators):
        with compss_open(args.path_out + 'tree_' + str(i)):
            pass

    open_time = time.time()
    print('Open time: ' + str(open_time - fit_time))

    y_predicted = forest.predict()
    if y_predicted is not None:
        y_real = read_csv(args.path_in + 'y_test.dat', header=None, dtype=object,
                          squeeze=True)
        accuracy = (np.count_nonzero(y_predicted == y_real))/len(y_real)

        predict_time = time.time()
        print('Predict time: ' + str(predict_time - open_time))
        print 'Accuracy: ', accuracy


if __name__ == "__main__":
    main()
