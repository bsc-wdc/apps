import numpy as np
import pandas
import scipy as sp
import h5py
from pandas import read_csv, DataFrame
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task

DATA_PATH = '/home/bscuser/datasets/'


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


@task(returns=pandas.DataFrame)
def to_dataframe(path):
    return read_csv(path, sep=' ', header=None, squeeze=True)


class Dataset(object):
    """A class that represents the train and test data."""

    def __init__(self,
                 name='0',
                 prediction_type='class',
                 path=DATA_PATH):
        self.name = name
        self.prediction_type = prediction_type
        self.path = path

    def read_and_wait_all(self):
        dataset_path = self.path + self.prediction_type + '_' + self.name + '_'
        X_train = to_dataframe(dataset_path + 'train_X.dat')
        y_train = to_dataframe(dataset_path + 'train_y.dat')
        X_test = to_dataframe(dataset_path + 'test_X.dat')
        y_test = None
        # y_test = to_dataframe(dataset_path + 'test_y.dat')
        X_train = compss_wait_on(X_train)
        y_train = compss_wait_on(y_train)
        X_test = compss_wait_on(X_test)
        # y_test  = compss_wait_on(y_test)
        return X_train, y_train, X_test, y_test

    def read(self, part_name):
        dataset_path = self.path + self.prediction_type + '_' + self.name + '_'
        return to_dataframe(dataset_path + part_name + '.dat')

    def save(self, X_train, y_train, X_test, y_test, file_per_feature=False, hdf5=False):
        X_train_t = np.asarray(zip(*X_train))
        X_test_t = np.asarray(zip(*X_test))
        dataset_path = self.path
        if not file_per_feature:
            dataset_path = dataset_path + self.prediction_type + '_' + self.name + '_'
            if hdf5:
                filename = dataset_path[:-1]+'.hdf5'
                DataFrame(X_train).to_hdf(filename, "train_X", mode='w', format='table')
                DataFrame(X_test).to_hdf(filename, "test_X",  format='table')
                # Not working because of a limit on number of columns of PyTables table
                # DataFrame(X_train_t).to_hdf(filename, "train_X_features", format='table')
                # DataFrame(X_test_t).to_hdf(filename, "test_X_features", format='table')
                DataFrame(y_train).to_hdf(filename, "train_y", format='table')
                DataFrame(y_test).to_hdf(filename, "test_y", format='table')
            else:
                np.savetxt(dataset_path + 'train_X.dat', X_train)
                np.savetxt(dataset_path + 'test_X.dat', X_test)
                if self.prediction_type == 'regr':
                    np.savetxt(dataset_path + 'train_y.dat', y_train)
                    np.savetxt(dataset_path + 'test_y.dat', y_test)
                else:
                    np.savetxt(dataset_path + 'train_y.dat', y_train, fmt='%s')
                    np.savetxt(dataset_path + 'test_y.dat', y_test, fmt='%s')
        else:
            X_train_transposed = zip(*X_train)
            for i in range(len(X_train_transposed)):
                np.savetxt(dataset_path + 'x_' + str(i) + '.dat', X_train_transposed[i])
            if self.prediction_type == 'regr':
                np.savetxt(dataset_path + 'y.dat', y_train)
            else:
                np.savetxt(dataset_path + 'y.dat', y_train, fmt='%s')
