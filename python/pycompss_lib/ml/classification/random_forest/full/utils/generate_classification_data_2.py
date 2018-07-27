import argparse
from distutils import util
import numpy as np

from sklearn.datasets import make_classification
from random import randint
import sys
import running_utils

if __name__ == "__main__":
    X_train = np.empty(shape=(200, 10), dtype=np.float32)
    y_train = np.empty(shape=(200,), dtype=np.float32)
    X_test = np.empty(shape=(200, 10), dtype=np.float32)
    y_test = np.empty(shape=(200,), dtype=np.float32)
    [X, y] = make_classification(
        n_samples=400,
        n_features=10,
        n_classes=3,
        n_informative=7,
        n_redundant=1,
        n_repeated=1,
        n_clusters_per_class=2,
        shuffle=True,
        random_state=0)
    X_train[:] = X[0:200].astype(dtype=np.float32)
    y_train[:] = y[0:200].astype(dtype=np.float32)
    X_test[:] = X[200:400].astype(dtype=np.float32)
    y_test[:] = y[200:400].astype(dtype=np.float32)
    ds_kwargs = {'name': 'dt_test_8', 'path': '/home/bscuser/datasets/dt_test_8/', 'prediction_type': 'class'}
    ds = running_utils.Dataset(**ds_kwargs)
    ds.save(X_train, y_train, X_test, y_test, False, False, True)
