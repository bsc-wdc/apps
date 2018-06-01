#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task import task


@task(returns=int)
def sort_partition(path):
    """ Sorts data, which is assumed to consists of (key, value) tuples list.
    :param path: file absolute path where the list of tuples to be sorted is located.
    :return: sorted list of tuples.
    """
    import pickle
    import numpy as np
    f = open(path, 'r')
    iterator = pickle.load(f)
    f.close()
    res = np.sort(iterator, kind='mergesort')
    return len(res)


def sort_by_key(files_paths):
    """ Sort by key.
    :param files_paths: List of paths of the input files.
    :return: Length of the list of elements sorted.
    """
    from pycompss.api.api import compss_wait_on
    n = list(map(sort_partition, files_paths))
    n = compss_wait_on(n)
    return len(n)


def main():
    import sys
    import os
    import time
    path = sys.argv[1]

    files_paths = []
    for f in os.listdir(path):
        files_paths.append(path + '/' + f)
    start_time = time.time()
    result = sort_by_key(files_paths)

    print("Elapsed Time(s)")
    print(time.time() - start_time)
    print(result)


if __name__ == "__main__":
    main()
