import argparse
from distutils import util
import numpy as np

from sklearn.datasets import make_classification
from random import randint
import sys
import running_utils

if __name__ == "__main__":
    data_path = running_utils.DATA_PATH
    parser = argparse.ArgumentParser(description='Generate samples for a regression.')
    parser.add_argument('--n_samples', type=int, default=1000, help="The number of samples.")
    parser.add_argument('--n_features', type=int, default=100, help="The number of features.")
    parser.add_argument('--n_classes', type=int, default=3, help="The number of classes.")
    parser.add_argument('--name', default='default')
    parser.add_argument('--path')
    parser.add_argument('--file_per_feature', default=False, type=util.strtobool,
                        help='Whether to use separate output files for each feature.')
    parser.add_argument('--hdf5', default=False, type=util.strtobool, help='Whether to use hdf5 file format.')
    parser.add_argument('--npy', default=False, type=util.strtobool, help='x.npy, x_t.npy and y.dat')
    parser.add_argument('--seed', default=-1, type=int,
                        help='Seed for the random number generator. If negative, the random number generator \
                        is the RandomState instance used by `np.random`.')
    args = parser.parse_args()
    if len(sys.argv) > 1:
        dataset_id = sys.argv[1]
    seed = args.seed if args.seed >= 0 else None
    created_samples = 0
    chunk_size = np.int64(1000000)*np.int64(100)
    max_sample_chunk_size = chunk_size//args.n_features
    X_train = np.empty(shape=(args.n_samples, args.n_features), dtype=np.float32)
    y_train = np.empty(shape=(args.n_samples,), dtype=np.float32)
    X_test = np.empty(shape=(args.n_samples, args.n_features), dtype=np.float32)
    y_test = np.empty(shape=(args.n_samples,), dtype=np.float32)
    while created_samples < args.n_samples:
        sample_chunk_size = min(max_sample_chunk_size, args.n_samples - created_samples)
        [X_train_chunk, y_train_chunk] = make_classification(
            n_samples=sample_chunk_size,
            n_features=args.n_features,
            n_classes=args.n_classes,
            n_informative=args.n_features // 3,
            n_redundant=2,
            n_repeated=1,
            n_clusters_per_class=2,
            shuffle=True,
            random_state=seed)
        X_train[created_samples:created_samples+sample_chunk_size] = X_train_chunk.astype(dtype=np.float32)
        y_train[created_samples:created_samples+sample_chunk_size] = y_train_chunk.astype(dtype=np.float32)
        seed = None if seed is None else seed+1
        [X_test_chunk, y_test_chunk] = make_classification(
            n_samples=sample_chunk_size,
            n_features=args.n_features,
            n_classes=args.n_classes,
            n_informative=args.n_features // 3,
            n_redundant=2,
            n_repeated=1,
            n_clusters_per_class=2,
            shuffle=True,
            random_state=seed if seed is None else seed + 1)
        X_test[created_samples:created_samples+sample_chunk_size] = X_test_chunk.astype(dtype=np.float32)
        y_test[created_samples:created_samples+sample_chunk_size] = y_test_chunk.astype(dtype=np.float32)
        seed = None if seed is None else seed + 1
        created_samples += sample_chunk_size
    ds_kwargs = {k: v for k, v in vars(args).items() if k in ('name', 'path') and v is not None}
    ds_kwargs['prediction_type'] = 'class'
    ds = running_utils.Dataset(**ds_kwargs)
    ds.save(X_train, y_train, X_test, y_test, args.file_per_feature, args.hdf5, args.npy)
