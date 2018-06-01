#!/usr/bin/python
# -*- coding: utf-8 -*-
from pycompss.api.task import task


@task(returns=int)
def sortPartition(path):
    """ Sorts data, which is assumed to consists of (key, value) tuples list.
    :param path: file absolute path where the list of tuples to be sorted is located.
    :return: sorted list of tuples.
    """
    import pickle
    f = open(path, 'r')
    iterator = pickle.load(f)
    f.close()
    res = np.sort(iterator, kind='mergesort')
    return len(res)


def sortByKey(files_paths, n_times):
    """ Sort by key.
    :param files_paths: List of paths of the input files.
    :param n_times: Number of times to do the sort by key.
    :return: Length of the list of elements sorted.
    """
    from pycompss.api.api import compss_wait_on
    fo_list = []
    for i in range(n_times):
        fo_list.append(list(map(sortPartition, files_paths)))
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
