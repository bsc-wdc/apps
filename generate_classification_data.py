import argparse

from sklearn.datasets import make_classification
from random import randint
import sys

import utils

if __name__ == "__main__":
    data_path = utils.DATA_PATH
    parser = argparse.ArgumentParser(description='Generate samples for a regression.')
    parser.add_argument('--n_samples', type=int, default=1000, help="The number of samples.")
    parser.add_argument('--n_features', type=int, default=100, help="The number of features.")
    parser.add_argument('--n_classes', type=int, default=3, help="The number of classes.")
    parser.add_argument('--name', default='default')
    parser.add_argument('--path')
    args = parser.parse_args()
    if len(sys.argv) > 1:
        dataset_id = sys.argv[1]
    seed = randint(0, 999999)
    [X, y] = make_classification(
        n_samples=2*args.n_samples,
        n_features=args.n_features,
        n_classes=args.n_classes,
        n_informative=args.n_features // 3,
        n_redundant=2,
        n_repeated=1,
        n_clusters_per_class=2,
        shuffle=True,
        random_state=seed)
    X_train = X[:len(X) / 2]
    y_train = y[:len(y) / 2]
    X_test = X[len(X) / 2:]
    y_test = y[len(y) / 2:]
    ds_kwargs = {k: v for k, v in vars(args).iteritems() if k in ('name', 'path') and v is not None}
    ds_kwargs['prediction_type'] = 'class'
    ds = utils.Dataset(**ds_kwargs)
    ds.save(X_train, y_train, X_test, y_test)
