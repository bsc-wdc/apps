#!/usr/bin/python
# -*- coding: utf-8 -*-
from pycompss.api.task import task


@task(returns=list)
def sortPartition(path):
    """Sorts data, which is assumed to consists of (key, value) pairs"""
    import pickle
    import operator
    #import numpy as ynp
    f = open(path, 'r')
    data = pickle.load(f)
    f.close()
    #res = sorted(data, key=lambda (k, v): k, reverse=not ascending)
    res = sorted(data.items(), key=operator.itemgetter(1), reverse=False)
    #res = np.sort(data, kind='mergesort')
    return res


def sortByKey(paths):
    from pycompss.api.api import compss_wait_on
    n = map(sortPartition, paths)
    res = merge_reduce(reducetask, n)
    res = compss_wait_on(res)
    return res


@task(returns=list, priority=True)
def reducetask(a, b):
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
    if i < len(a):
        res.append(a[i:])
    elif j < len(b):
        res.append(b[j:])

    return res


def merge_reduce(function, data):
    import Queue
    q = Queue.Queue()
    for i in data:
        q.put(i)
    while not q.empty():
        x = q.get()
        if not q.empty():
            y = q.get()
            q.put(function(x, y))
        else:
            return x

if __name__ == "__main__":
    import sys
    import os
    import time
    path = sys.argv[1]

    X = []
    for file in os.listdir(path):
        X.append(path+'/'+file)
    startTime = time.time()
    result = sortByKey(X)

    print "Ellapsed Time(s)"
    print time.time()-startTime
    print result
