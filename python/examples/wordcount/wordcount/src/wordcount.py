#!/usr/bin/python
#
#  Copyright 2002-2019 Barcelona Supercomputing Center (www.bsc.es)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# -*- coding: utf-8 -*-

"""wordcount self read"""

import sys
import os
import time
from pycompss.api.task import task
from pycompss.api.parameter import *
from collections import defaultdict


@task(returns=dict, file=FILE_IN, priority=True)
def wordcount(file):
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


@task(returns=dict, path_file=FILE_IN, priority=True)
def wordcount_block(path_file, start, block_size):
    """ Perform a wordcount of a portion of a file.
    :param path_file: Absolute path of the file to process.
    :param start: wordcount starting point.
    :param block_size: Block to process size in bytes.
    :return: dictionary with the appearance of each word.
    """
    fp = open(path_file)
    fp.seek(start)
    block = fp.read(block_size)
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
    for k, v in dic1.items():
        dic2[k] += v
    return dic2


def merge_reduce(f, data):
    """ Apply function cumulatively to the items of data,
        from left to right in binary tree structure, so as to
        reduce the data to a single value.
    :param f: function to apply to reduce data
    :param data: List of items to be reduced
    :return: result of reduce the data to a single value
    """
    from collections import deque
    q = deque(range(len(data)))
    while len(q):
        x = q.popleft()
        if len(q):
            y = q.popleft()
            data[x] = f(data[x], data[y])
            q.append(x)
        else:
            return data[x]


def main():
    from pycompss.api.api import compss_wait_on

    path_file = sys.argv[1]
    multipleFiles = sys.argv[2]
    if multipleFiles == "False":
        # All text is in a unique text file (need block size).
        block_size = int(sys.argv[3])

    print("wordcount:")
    print("Path: %s" % path_file)
    print("Multiple files?: %s" % multipleFiles)
    if multipleFiles == "False":
        print("Block size: %d" % block_size)

    start = time.time()

    partial_result = []
    if multipleFiles == "False":
        # Process only one file.
        # Tasks will process chunks of the file.
        data = open(path_file)
        data.seek(0, 2)
        file_size = data.tell()
        ind = 0
        while ind < file_size:
            partial_result.append(wordcount_block(path_file, ind, block_size))
            ind += block_size
    else:
        # Process multiple files.
        # Each file will be processed by a task.
        for fileName in os.listdir(path_file):
            partial_result.append(wordcount(path_file + fileName))

    result = merge_reduce(reduce, partial_result)

    result = compss_wait_on(result)
    elapsed = time.time()-start

    print("Elapsed Time: ", elapsed)
    # print result


if __name__ == "__main__":
    main()
