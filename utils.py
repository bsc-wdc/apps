import numpy as np
import scipy as sp
from pandas import read_csv

DATA_PATH = './datasets/'


def are_equal(a, b):
    if isinstance(a, (np.ndarray, list, tuple)):
        for i in range(len(a)):
            if not are_equal(a[i], b[i]):
                return False
        return True
    elif isinstance(a, sp.sparse.csr.csr_matrix):
        return (a != b).nnz == 0
    elif isinstance(a, np.float64):
        return abs(a - b) < 1e-10
    else:
        return a == b or np.isnan(a) and np.isnan(b)


class Dataset(object):
    """A class that represents the train and test data."""

    def __init__(self,
                 name='0',
                 prediction_type='class',
                 path=DATA_PATH):
        self.name = name
        self.prediction_type = prediction_type
        self.path = path

    def read(self):
        dataset_path = self.path + self.prediction_type + '_' + self.name + '_'
        X_train = read_csv(dataset_path + 'train_X.dat', sep=' ', header=None)
        y_train = np.ravel(read_csv(dataset_path + 'train_y.dat', sep=' ', header=None))
        X_test = read_csv(dataset_path + 'test_X.dat', sep=' ', header=None)
        y_test = None
        # y_test = np.ravel(read_csv(dataset_path + 'test_y.dat', sep=' ', header=None))
        return X_train, y_train, X_test, y_test

    def save(self, X_train, y_train, X_test, y_test):
        dataset_path = self.path + self.prediction_type + '_' + self.name + '_'
        np.savetxt(dataset_path + 'train_X.dat', X_train)
        np.savetxt(dataset_path + 'test_X.dat', X_test)
        if self.prediction_type == 'regr':
            np.savetxt(dataset_path + 'train_y.dat', y_train)
            np.savetxt(dataset_path + 'test_y.dat', y_test)
        else:
            np.savetxt(dataset_path + 'train_y.dat', y_train, fmt='%s')
            np.savetxt(dataset_path + 'test_y.dat', y_test, fmt='%s')
