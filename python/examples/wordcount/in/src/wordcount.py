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

"""Wordcount self read"""

import sys
import pickle
import time
from pycompss.api.task import task
from functools import reduce


@task(returns=dict, priority=True)
def wordcount_self_read(path_file, start, size_block):
    fp = open(path_file)
    fp.seek(start)
    aux = fp.read(size_block)
    fp.close()
    data = aux.strip().split(" ")
    partial_result = {}
    for entry in data:
        if entry not in partial_result:
            partial_result[entry] = 1
        else:
            partial_result[entry] = partial_result[entry] + 1
    return partial_result


@task(returns=dict)
def reduce(dic1, dic2):
    for k in dic2:
        if k in dic1:
            dic1[k] = dic1[k] + dic2[k]
        else:
            dic1[k] = dic2[k]
    return dic1


def merge_reduce(partial_result):
    n = len(partial_result)
    act = [j for j in range(n)]
    while n > 1:
        aux = []
        if n % 2:
            partial_result[act[len(act)-2]] = reduce(partial_result[act[len(act)-2]], partial_result[act[len(act)-1]])
            act.pop(len(act)-1)
            n = n-1
        for i in range(0, n, 2):
            partial_result[act[i]] = reduce(partial_result[act[i]], partial_result[act[i+1]])
            aux.append(act[i])
        act = aux
        n = len(act)
    return partial_result[0]


def main():
    from pycompss.api.api import compss_wait_on

    path_file = sys.argv[1]
    result_file = sys.argv[2]
    size_block = int(sys.argv[3])

    start = time.time()
    data = open(path_file)
    data.seek(0, 2)
    file_size = data.tell()
    ind = 0
    partial_result = []
    while ind < file_size:
        partial_result.append(wordcount_self_read(path_file, ind, size_block))
        ind += size_block
    result = merge_reduce(partial_result)
    result = compss_wait_on(result)

    print("Elapsed Time")
    print(time.time()-start)

    aux = list(result.items())
    ff = open(result_file, 'w')
    pickle.dump(aux, ff)
    ff.close()


if __name__ == "__main__":
    main()
