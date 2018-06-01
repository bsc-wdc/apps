#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from pycompss.api.parameter import *


@task(file=FILE_INOUT)
def readData(path):
    """
    Read the data from a given file.
    :param path: Input file path.
    :return: Content of the input file.
    """
    f = open(path, 'r')
    import pickle
    data = pickle.load(f)
    f.close()
    return data


@task(file=FILE_IN, returns=int)
def sortPartition(data):
    """ Sorts data, which is assumed to consists of (key, value) tuples list.
    :param data: List of tuples to be sorted.
    :return: sorted list of tuples.
    """
    res = np.sort(data, kind='mergesort')
    return len(res)


def sortByKey(files_paths, n_times):
    """ Sort by key.
    :param files_paths: List of paths of the input files.
    :param n_times: Number of times to do the sort by key.
    :return: Length of the list of elements sorted.
    """
    from pycompss.api.api import compss_wait_on
    fo_list = []
    dataset = list(map(readData, files_paths))
    for i in range(n_times):
        fo_list.append(list(map(sortPartition, dataset)))
    result_list = compss_wait_on(fo_list)
    return len(result_list)


def main():
    import sys
    import os
    import time
    path = sys.argv[1]
    n_times = int(sys.argv[2])

    files_paths = []
    for f in os.listdir(path):
        files_paths.append(path + '/' + f)

    startTime = time.time()
    result = sortByKey(files_paths, n_times)
    endTime = time.time() - startTime

    print("Elapsed Time(s)")
    print(endTime)
    print(result)


if __name__ == "__main__":
    main()
