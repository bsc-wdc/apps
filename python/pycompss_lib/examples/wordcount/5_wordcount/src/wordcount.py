#!/usr/bin/python
# -*- coding: utf-8 -*-
'''Wordcount self read'''
from pycompss.api.task import task
from pycompss.api.parameter import *
from collections import defaultdict


@task(returns=dict, file=FILE_IN, priority=True)
def wordCount(file):
    """ Perform a wordcount of a file.
    :param fle: Absolute path of the file to process.
    :return: dictionary with the appearance of each word.
    """
    fp = open(file)
    data = fp.read().split(" ")
    fp.close()
    result = defaultdict(int)
    for word in data:
        result[word] += 1
    return result


@task(returns=dict, pathFile=FILE_IN, priority=True)
def wordCountBlock(pathFile, start, sizeBlock):
    """ Perform a wordcount of a portion of a file.
    :param pathFile: Absolute path of the file to process.
    :param start: Wordcount starting point.
    :param sizeBlock: Block to process size in bytes.
    :return: dictionary with the appearance of each word.
    """
    fp = open(pathFile)
    fp.seek(start)
    block = fp.read(sizeBlock)
    fp.close()
    data = block.strip().split(" ")
    result = defaultdict(int)
    for word in data:
        result[word] += 1
    return result


@task(returns=dict)
def reduce(dic1, dic2):
    """ Reduce dictionaries a and b.
    :param a: dictionary.
    :param b: dictionary.
    :return: dictionary result of merging a and b.
    """
    for k, v in dic1.iteritems():
        dic2[k] += v
    # This is destructive regarding dic2, but the algorithm
    # knows that it becomes irrelevant.
    return dic2


def mergeReduce(function, data):
    """ Apply function cumulatively to the items of data,
        from left to right in binary tree structure, so as to
        reduce the data to a single value.
    :param function: function to apply to reduce data
    :param data: List of items to be reduced
    :return: result of reduce the data to a single value
    """
    from collections import deque
    q = deque(xrange(len(data)))
    while len(q):
        x = q.popleft()
        if len(q):
            y = q.popleft()
            data[x] = function(data[x], data[y])
            q.append(x)
        else:
            return data[x]


if __name__ == "__main__":
    import sys
    import os
    import time
    pathFile = sys.argv[1]
    multipleFiles = sys.argv[2]
    if multipleFiles == "False":
        # All text is in a unique text file (need block size).
        sizeBlock = int(sys.argv[3])

    print "Wordcount:"
    print "Path: %s" % (pathFile)
    print "Multiple files?: %s" % (multipleFiles)
    if multipleFiles == "False":
        print "Block size: %d" % (sizeBlock)

    start = time.time()
    from pycompss.api.api import compss_wait_on

    partialResult = []

    if multipleFiles == "False":
        # Process only one file.
        # Tasks will process chunks of the file.
        data = open(pathFile)
        data.seek(0, 2)
        file_size = data.tell()
        ind = 0
        while ind < file_size:
            partialResult.append(wordCountBlock(pathFile, ind, sizeBlock))
            ind += sizeBlock
    else:
        # Process multiple files.
        # Each file will be processed by a task.
        for fileName in os.listdir(pathFile):
            partialResult.append(wordCount(pathFile + fileName))

    result = mergeReduce(reduce, partialResult)

    result = compss_wait_on(result)
    elapsed = time.time()-start

    print "Elapsed Time: ", elapsed
    # print result
