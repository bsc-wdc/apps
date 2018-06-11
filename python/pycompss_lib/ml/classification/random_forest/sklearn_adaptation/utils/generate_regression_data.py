import argparse

from sklearn.datasets import make_regression
from random import randint

import python.pycompss_lib.ml.classification.random_forest.sklearn_adaptation.utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate samples for a regression.')
    parser.add_argument('--n_samples', type=int, default=1000, help="The number of samples.")
    parser.add_argument('--n_features', type=int, default=100, help="The number of features.")
    parser.add_argument('--name', default='default')
    parser.add_argument('--path')
    args = parser.parse_args()
    seed = randint(0, 999999)
    [X, y] = make_regression(
        n_samples=2*args.n_samples,
        n_features=args.n_features,
        n_informative=80,
        n_targets=1,
        bias=0.5,
        effective_rank=3,
        tail_strength=0.5,
        noise=0.8,
        shuffle=True,
        coef=False,
        random_state=seed)
    X_train = X[:len(X) / 2]
    y_train = y[:len(y) / 2]
    X_test = X[len(X) / 2:]
    y_test = y[len(y) / 2:]
    ds_kwargs = {k: v for k, v in vars(args).iteritems() if k in ('name', 'path') and v is not None}
    ds_kwargs['prediction_type'] = 'regr'
    ds = python.pycompss_lib.ml.classification.random_forest.sklearn_adaptation.utils.Dataset(**ds_kwargs)
    ds.save(X_train, y_train, X_test, y_test)
