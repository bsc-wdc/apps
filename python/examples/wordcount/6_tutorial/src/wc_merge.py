#!/usr/bin/python
#
#  Copyright 2002-2018 Barcelona Supercomputing Center (www.bsc.es)
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

import sys
import os
from pycompss.api.task import task
from pycompss.api.parameter import *


@task(file_path=FILE_IN, returns=list)
def read_file(file_path):
    """ Read a file and return a list of words.
    :param file_path: file's path
    :return: list of words
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data += line.split()
    return data


@task(returns=dict)
def wordcount(data):
    """ Construct a frequency word dictionary from a list of words.
    :param data: a list of words
    :return: a dictionary where key=word and value=#appearances
    """
    partial_result = {}
    for entry in data:
        if entry in partial_result:
            partial_result[entry] += 1
        else:
            partial_result[entry] = 1
    return partial_result


@task(returns=dict, priority=True)
def merge_two_dicts(dic1, dic2):
    """ Update a dictionary with another dictionary.
    :param dic1: first dictionary
    :param dic2: second dictionary
    :return: dic1+=dic2
    """
    for k in dic2:
        if k in dic1:
            dic1[k] += dic2[k]
        else:
            dic1[k] = dic2[k]
    return dic1


def merge_reduce(f, data):
    """ Apply function cumulatively to the items of data,
        from left to right in binary tree structure, so as to
        reduce the data to a single value.
    :param f: function to apply to reduce data
    :param data: List of items to be reduced
    :return: result of reduce the data to a single value
    """
    from collections import deque
    q = deque(list(range(len(data))))
    while len(q):
        x = q.popleft()
        if len(q):
            y = q.popleft()
            data[x] = f(data[x], data[y])
            q.append(x)
        else:
            return data[x]


if __name__ == "__main__":
    from pycompss.api.api import compss_wait_on

    # Get the dataset path
    dataset_path = sys.argv[1]

    # Construct a list with the file's paths from the dataset
    paths = []
    for fileName in os.listdir(dataset_path):
        paths.append(os.path.join(dataset_path, fileName))

    # Read file's content execute a wordcount on each of them
    partial_result = []
    for p in paths:
        data = read_file(p)
        partial_result.append(wordcount(data))

    # Accumulate the partial results to get the final result.
    result = merge_reduce(merge_two_dicts, partial_result)

    # Wait for result
    result = compss_wait_on(result)

    print("Result:")
    from pprint import pprint
    pprint(result)
    print("Words: {}".format(sum(result.values())))
