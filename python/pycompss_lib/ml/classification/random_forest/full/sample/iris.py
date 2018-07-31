from __future__ import division
import argparse
import os
import time
import numpy as np

from pycompss.api.api import compss_barrier
from pycompss.api.parameter import DIRECTION
from pycompss.runtime.binding import get_file
from sklearn.datasets import load_iris

from forest import RandomForestClassifier
from pandas import read_csv


def main():
    initial_time = time.time()
    parser = argparse.ArgumentParser(description='Random forest classifier.')

    # RandomForestClassifier params
    parser.add_argument('--n_estimators', type=int, help='Number of trees to build.')
    parser.add_argument('--max_depth', type=int, default=None, help='Depth of the decision tree.')
    parser.add_argument('--distr_depth', type=int, default=None, help='Tasks are distributed up to this depth.')
    parser.add_argument('--try_features', default=None, help='Number of features to try at each split.')

    args = parser.parse_args()

    x, y = load_iris(return_X_y=True)
    x_train = x[::2]
    y_train = y[::2]
    x_test = x[1::2]
    y_test = y[1::2]
    np.save('x.npy', x_train.astype(np.float32))
    np.save('x_t.npy', x_train.astype(np.float32).T.copy(order='C'))
    np.savetxt('y.dat', y_train, fmt='%s')
    np.save('x_test.npy', x_test)

    forest = RandomForestClassifier(os.getcwd(), 75, 4, os.getcwd(),
                                    args.n_estimators, args.max_depth, args.distr_depth, args.try_features)
    forest.fit()

    compss_barrier()

    fit_time = time.time()

    print(args)
    print('Fit time: ' + str(fit_time - initial_time))

    for tree in forest.trees:
        tree_file = get_file(os.path.join(tree.path_out, tree.name_out), DIRECTION.IN)
        tree.path_out, tree.name_out = os.path.split(tree_file)

    get_files_time = time.time()
    print('Open time: ' + str(get_files_time - fit_time))

    predict_start_time = time.time()
    y_predicted = forest.predict()
    predict_time = time.time()
    accuracy = (np.count_nonzero(y_predicted == [str(num) for num in y_test])) / len(y_test)

    print('Predict time: ' + str(predict_time - predict_start_time))
    print 'Accuracy: ', accuracy


if __name__ == "__main__":
    main()
