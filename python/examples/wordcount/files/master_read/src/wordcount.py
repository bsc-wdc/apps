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
import pickle
import time
from pycompss.api.task import task


@task(returns=dict)
def wordcount(data):
    partial_result = {}
    for entry in data:
        for word in entry:
            if word not in partial_result:
                partial_result[word] = 1
            else:
                partial_result[word] += 1
    return partial_result


@task(returns=dict)
def reduce(dic1, dic2):
    for k in dic2:
        if k in dic1:
            dic1[k] += dic2[k]
        else:
            dic1[k] = dic2[k]
    return dic1


def merge_reduce(f, data):
    import queue
    q = queue.Queue()
    for i in data:
        q.put(i)
    while not q.empty():
        x = q.get()
        if not q.empty():
            y = q.get()
            q.put(f(x, y))
        else:
            return x


def main():
    from pycompss.api.api import compss_wait_on

    path = sys.argv[1]
    result_file = sys.argv[2]

    partial_result = []

    start = time.time()

    for file in os.listdir(path):
        f = open(path + file)
        data = f.readlines()
        partial_result.append(wordcount(data))

    print("All wordcount tasks submitted")

    result = merge_reduce(reduce, partial_result)
    result = compss_wait_on(result)

    print("Elapsed time: ")
    print(time.time() - start)

    aux = list(result.items())
    ff = open(result_file, 'w')
    pickle.dump(aux, ff)
    ff.close()


if __name__ == "__main__":
    main()
