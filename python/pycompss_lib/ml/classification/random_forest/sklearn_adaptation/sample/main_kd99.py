import argparse
import time

from distutils import util

from pandas import read_csv
from pycompss.api.api import compss_barrier

from python.pycompss_lib.ml.classification.random_forest.sklearn_adaptation.src import forest
import sklearn as sk

def main():
    initial_time = time.time()
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS,
                                     description='Predict regression value using a decision tree regressor.')
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

    rf_kwargs = {k: v for k, v in vars(args).items() if k not in ('name', 'path', 'regr', 'sklearn')}

    time_1 = time.time()

    df = read_csv(args.path + args.name, header=None, squeeze=True)
    time_2 = time.time()

    y_train = df[df.columns[-1]]
    X_train = df.drop(labels=df.columns[-1], axis=1)

    time_3 = time.time()

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
    # print(type(rf))

    rf.fit(X_train, y_train)

    compss_barrier()

    time_4 = time.time()

    print('X_shape: ' + str(X_train.shape))
    print('y_shape: ' + str(y_train.shape))
    print('Time 1: ' + str(time_1-initial_time))
    print('Time 2: ' + str(time_2-initial_time))
    print('Time 3: ' + str(time_3 - initial_time))
    print('Time 4: ' + str(time_4 - initial_time))


if __name__ == "__main__":
    main()
