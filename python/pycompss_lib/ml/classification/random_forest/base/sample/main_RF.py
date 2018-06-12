import argparse
import time

from distutils import util

from pycompss.api.api import compss_wait_on

import running_utils
import forest
import sklearn as sk


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS,
                                     description='Predict regression value using a random src.')
    # RandomForest params
    parser.add_argument('--n_estimators', type=int, help='The number of trees in the src.')
    parser.add_argument('--bootstrap', type=util.strtobool,
                        help='0 or 1. Whether bootstrap samples are used when building trees.')
    parser.add_argument('--oob_score', type=util.strtobool,
                        help='0 or 1. Whether to use out-of-bag samples to estimate the R^2 on unseen data.')
    parser.add_argument('--random_state', help='int, RandomState instance or None, optional (default=None)')
    parser.add_argument('--verbose', help='int (default=0): Controls the verbosity of the tree building process.')
    parser.add_argument('--warm_start', type=util.strtobool,
                        help='If True, reuse the solution of the previous call to fit.')

    # DecisionTree params:
    parser.add_argument('--max_features', help='The number of features to consider when looking for the best split')
    parser.add_argument('--criterion', help='The function to measure the quality of a split.')
    parser.add_argument('--splitter', help='The strategy used to choose the split at each node.')
    parser.add_argument('--max_depth', type=int, help='The maximum depth of the tree.')
    parser.add_argument('--min_samples_split', type=float,
                        help='The minimum number of samples required to split an internal node.')

    # Execution params
    parser.add_argument('--name', help='Dataset identifier.')
    parser.add_argument('--path', help='Path to dataset folder.')
    parser.add_argument('--regr', type=util.strtobool, default=False,
                        help='0 or 1. Use regression instead of classification.')
    parser.add_argument('--sklearn', type=util.strtobool, default=False,
                        help='0 or 1. Use sklearn implementation instead of COMPSs.')

    args = parser.parse_args()

    ds_kwargs = {k: v for k, v in vars(args).items() if k in ('name', 'path')}
    if args.regr:
        ds_kwargs['prediction_type'] = 'regr'
    else:
        ds_kwargs['prediction_type'] = 'class'

    rf_kwargs = {k: v for k, v in vars(args).items() if k not in ('name', 'path', 'regr', 'sklearn')}

    # RandomForestRegressor Algorithm
    initial_time = time.time()

    ds = running_utils.Dataset(**ds_kwargs)

    X_train = ds.read('train_X')
    y_train = ds.read('train_y')
    X_test = ds.read('test_X')

    if args.sklearn:
        if args.regr:
            rf = sk.ensemble.RandomForestRegressor(**rf_kwargs)
        else:
            rf = sk.ensemble.RandomForestClassifier(**rf_kwargs)
    else:
        if args.regr:
            rf = forest.RandomForestRegressor(**rf_kwargs)
        else:
            rf = forest.RandomForestClassifier(**rf_kwargs)

    time_0 = time.time()

    X_train = compss_wait_on(X_train)
    y_train = compss_wait_on(y_train)

    rf.fit(X_train, y_train)

    time_1 = time.time()

    # compss_barrier()

    X_test = compss_wait_on(X_test)

    rf.predict(X_test)

    time_2 = time.time()

    print('X_shape: ' + str(X_train.shape))
    print('y_shape: ' + str(y_train.shape))
    print('Load time: ' + str(time_0-initial_time))
    print('Load + Fit time: ' + str(time_1-initial_time))
    print('Load + Fit + Predict time: ' + str(time_2 - initial_time))


if __name__ == "__main__":
    main()
