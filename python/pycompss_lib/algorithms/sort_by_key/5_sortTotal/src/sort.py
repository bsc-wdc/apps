#!/usr/bin/python
# -*- coding: utf-8 -*-
from pycompss.api.task import task


@task(returns=list)
def sortPartition(path):
    """ Sorts data, which is assumed to consists of (key, value) tuples list.
    :param path: file absolute path where the list of tuples to be sorted is located.
    :return: sorted list of tuples.
    """
    import pickle
    import operator
    f = open(path, 'r')
    data = pickle.load(f)
    f.close()
    res = sorted(list(data.items()), key=operator.itemgetter(1), reverse=False)
    return res


def sortByKey(files_paths):
    """ Sort by key.
    :param files_paths: List of paths of the input files.
    :return: List of elements sorted.
    """
    from pycompss.api.api import compss_wait_on
    n = list(map(sortPartition, files_paths))
    res = merge_reduce(reducetask, n)
    res = compss_wait_on(res)
    return res


@task(returns=list, priority=True)
def reducetask(a, b):
    """ Reduce list a and list b.
        They must be already sorted.
    :param a: list.
    :param b: list.
    :return: result of merging a and b lists.
    """
    res = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            res.append(a[i])
            i += 1
        else:
            res.append(b[j])
            j += 1
    # Append the remaining tuples
    if i < len(a):
        res += a[i:]
    elif j < len(b):
        res += b[j:]
    return res


def merge_reduce(function, data):
    """ Apply function cumulatively to the items of data,
        from left to right in binary tree structure, so as to
        reduce the data to a single value.
    :param function: function to apply to reduce data.
    :param data: List of items to be reduced.
    :return: result of reduce the data to a single value.
    """
    import queue
    q = queue.Queue()
    for i in data:
        q.put(i)
    while not q.empty():
        x = q.get()
        if not q.empty():
            y = q.get()
            q.put(function(x, y))
        else:
            return x


def main():
    import sys
    import os
    import time
    path = sys.argv[1]

    files_paths = []
    for file in os.listdir(path):
        files_paths.append(path + '/' + file)
    startTime = time.time()
    result = sortByKey(files_paths)

    print("Elapsed Time(s)")
    print(time.time() - startTime)
    print(result)


if __name__ == "__main__":
    main()
