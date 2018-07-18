import argparse
from distutils import util
import numpy as np

from sklearn.datasets import make_classification
from random import randint
import sys
import running_utils

if __name__ == "__main__":
    X_train = np.empty(shape=(20000, 25), dtype=np.float32)
    y_train = np.empty(shape=(20000,), dtype=np.float32)
    X_test = np.empty(shape=(1000, 25), dtype=np.float32)
    y_test = np.empty(shape=(1000,), dtype=np.float32)
    [X, y] = make_classification(
        n_samples=21000,
        n_features=25,
        n_classes=3,
        n_informative=14,
        n_redundant=2,
        n_repeated=1,
        n_clusters_per_class=2,
        shuffle=True,
        random_state=0)
    X_train[:] = X[0:20000].astype(dtype=np.float32)
    y_train[:] = y[0:20000].astype(dtype=np.float32)
    X_test[:] = X[20000:21000].astype(dtype=np.float32)
    y_test[:] = y[20000:21000].astype(dtype=np.float32)
    ds_kwargs = {'name': 'dt_test_4', 'path': '/home/bscuser/datasets/dt_test_4/', 'prediction_type': 'class'}
    ds = running_utils.Dataset(**ds_kwargs)
    ds.save(X_train, y_train, X_test, y_test, False, False, True)
