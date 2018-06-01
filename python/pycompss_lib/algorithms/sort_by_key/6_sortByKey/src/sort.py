#!/usr/bin/python
# -*- coding: utf-8 -*-
from pycompss.api.task import task


@task(returns=list)
def sortPartitionFromFile(path):
    """ Sorts data, which is assumed to consists of (key, value) tuples list.
    :param path: file absolute path where the list of tuples to be sorted is located.
    :return: sorted list of tuples.
    """
    import pickle
    f = open(path, 'r')
    data = pickle.load(f)
    f.close()
    res = sorted(data, key=lambda tuple: tuple[0], reverse=False)
    return res


@task(returns=list)
def sortPartition(data):
    """ Sorts data, which is assumed to consists of (key, value) pairs list.
    :param data: list of tuples to be sorted by key.
    :return: sorted list of tuples.
    """
    res = sorted(data, key=lambda tuple: tuple[0], reverse=False)
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


def mergeReduce(function, data):
    """ Apply function cumulatively to the items of data,
        from left to right in binary tree structure, so as to
        reduce the data to a single value.
    :param function: function to apply to reduce data.
    :param data: List of items to be reduced.
    :return: result of reduce the data to a single value.
    """
    from collections import deque
    q = deque(range(len(data)))
    while len(q):
        x = q.popleft()
        if len(q):
            y = q.popleft()
            data[x] = function(data[x], data[y])
            q.append(x)
        else:
            return data[x]


@task(returns=list)
def generateFragment(numKeys, uniqueKeys, keyLength, uniqueValues, valueLength, randomSeed, hashFunction):
    """ Generate a fragment.
        Each fragment is list of pairs (K, V) generated randomly.
    :param numKeys: number of keys per fragment.
    :param uniqueKeys: number of unique keys.
    :param keyLength: length of each key.
    :param uniqueValues: number of unique values.
    :param valueLength: length of each value.
    :param randomSeed: Random seed.
    :param hashFunction: Boolean - use hash function.
    :return: fragment (list of tuples).
    """
    ints = generateIntData(numKeys, uniqueKeys, uniqueValues, randomSeed)
    data = [(paddedString(k_v[0], keyLength, hashFunction), paddedString(k_v[1], valueLength, hashFunction)) for k_v in ints]
    return data


def generateIntData(numKeys, uniqueKeys, uniqueValues, randomSeed):
    """ Generate a list of (int, int) tuples with random values.
    :param numKeys: number of keys per fragment.
    :param uniqueKeys: number of unique keys.
    :param uniqueValues: number of unique values.
    :param randomSeed: Random seed.
    :return: fragment (list of tuples).
    """
    import random
    random.seed(randomSeed)
    data = [(random.randint(0, uniqueKeys), random.randint(0, uniqueValues)) for i in range(numKeys)]
    return data


def paddedString(i, length, hashFunction):
    """ Converts a int to String with determined length.
    :param i: input integer.
    :param length: length (number of characters).
    :param hashFunction: Boolean - use hash function.
    :return: i as String.
    """
    fmtString = "{:0>" + str(length) + "d}"
    if hashFunction:
        out = hash(i)
        if len(str(out)) < length:
            return fmtString.format(out)
        else:
            return out
    else:
        return fmtString.format(i)


def main():
    import sys
    import os
    import time

    numKeys = int(sys.argv[1])
    uniqueKeys = int(sys.argv[2])
    keyLength = int(sys.argv[3])     # number of characters
    uniqueValues = int(sys.argv[4])
    valueLength = int(sys.argv[5])   # number of characters
    numFragments = int(sys.argv[6])
    keysPerFragment = numKeys / numFragments
    randomSeed = int(sys.argv[7])
    fromFiles = sys.argv[8]
    path = "Not used - Autogenerating dataset."
    if fromFiles == "true":
        # Ignore all parameters but 'path'
        # Each file represents a fragment
        # Each file has to contain a pickable dictionary {K,V} with the desired lengths and values.
        path = sys.argv[9]

    print("Sort by Key [(K,V)]:")
    print("Num keys: %d" % (numKeys))
    print("Unique keys: %d" % (uniqueKeys))
    print("Key length: %d" % (keyLength))
    print("Unique values: %d" % (uniqueValues))
    print("Value length: %d" % (valueLength))
    print("Num fragments: %d" % (numFragments))
    print("Keys per frameng: %d" % (keysPerFragment))
    print("Random seed: %d" % (randomSeed))
    print("From files: %r" % (fromFiles))
    print("Path: %s" % (path))

    startTime = time.time()
    from pycompss.api.api import compss_wait_on

    partialSorted = []
    result = []
    if fromFiles == "true":
        # Get Dataset from files (read within sortPartitionFromFile task)
        files = []
        for file in os.listdir(path):
            files.append(path+'/'+file)
        partialSorted = list(map(sortPartitionFromFile, files))
        result = mergeReduce(reducetask, partialSorted)
    else:
        # Autogenerate dataset
        for i in range(numFragments):
            fragment = generateFragment(keysPerFragment, uniqueKeys, keyLength, uniqueValues, valueLength, randomSeed, True)
            partialSorted.append(sortPartition(fragment))
            randomSeed += i
        result = mergeReduce(reducetask, partialSorted)

    result = compss_wait_on(result)

    print("Elapsed Time(s)")
    print(time.time()-startTime)
    print("Sorted by Key elements: %d" % (len(result)))
    # print result


if __name__ == "__main__":
    main()
