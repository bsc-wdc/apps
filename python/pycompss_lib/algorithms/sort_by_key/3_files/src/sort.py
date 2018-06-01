#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task import task


@task(returns=int)
def sortPartition(path, ascending=True):
    """ Sorts data, which is assumed to consists of (key, value) tuples list.
    :param path: file absolute path where the list of tuples to be sorted is located.
    :return: sorted list of tuples.
    """
    import pickle
    f = open(path, 'r')
    iterator = pickle.load(f)
    f.close()
    import operator
    res = sorted(iterator.items(), key=operator.itemgetter(1), reverse=not ascending)
    return len(res)


def sortByKey(files_paths):
    """ Sort by key.
    :param files_paths: List of paths of the input files.
    :return: List of elements sorted.
    """
    from pycompss.api.api import compss_wait_on
    n = map(sortPartition, files_paths)
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
    startTime = time.time()
    result = sortByKey(files_paths)

    print "Elapsed Time(s)"
    print time.time() - startTime
    print result


if __name__ == "__main__":
    main()
