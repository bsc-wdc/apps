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

"""wordcount Block divide master read"""

import sys
import pickle
import time
from pycompss.api.task import task
from pycompss.api.parameter import INOUT
from functools import reduce


def read_word(file_object):
    for line in file_object:
        for word in line.split():
            yield word


def read_word_by_word(fp, size_block):
    """Lazy function (generator) to read a file piece by piece in
    chunks of size approx size_block"""

    data = open(fp)
    block = []
    for word in read_word(data):
        block.append(word)
        if sys.getsizeof(block) > size_block:
            yield block
            block = []
    if block:
        yield block


@task(returns=dict)
def wordcount(data):
    partial_result = {}
    for entry in data:
        if entry not in partial_result:
            partial_result[entry] = 1
        else:
            partial_result[entry] += 1
    return partial_result


@task(dic1=INOUT)
def reduce(dic1, dic2):
    for k in dic2:
        if k in dic1:
            dic1[k] += dic2[k]
        else:
            dic1[k] = dic2[k]


def merge_reduce(partial_result):
    from pycompss.api.api import compss_wait_on
    n = len(partial_result)
    act = [j for j in range(n)]
    while n > 1:
        aux = []
        if n % 2:
            reduce(partial_result[act[len(act)-2]], partial_result[act[len(act)-1]])
            act.pop(len(act)-1)
            n -= 1
        for i in range(0, n, 2):
            reduce(partial_result[act[i]], partial_result[act[i+1]])
            aux.append(act[i])
        act = aux
        n = len(act)

    partial_result[0] = compss_wait_on(partial_result[0])
    return partial_result[0]


def main():
    path_file = sys.argv[1]
    result_file = sys.argv[2]
    size_block = int(sys.argv[3])

    start = time.time()
    result = merge_reduce(list(map(wordcount, read_word_by_word(path_file, size_block))))

    print("Elapsed Time")
    print(time.time()-start)

    aux = list(result.items())
    ff = open(result_file, 'w')
    pickle.dump(aux, ff)
    ff.close()


if __name__ == "__main__":
    main()
