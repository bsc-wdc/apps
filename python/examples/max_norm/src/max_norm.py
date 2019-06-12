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

from pycompss.api.task import task
from pycompss.api.parameter import *
from numpy import random


@task(returns=int)
def maxFinal(a, b):
    return max(a, b)


def mergeReduce(function, data):
    from collections import deque
    q = deque(list(range(0, len(data))))
    while len(q):
        x = q.popleft()
        if len(q):
            y = q.popleft()
            data[x] = function(data[x], data[y])
            q.append(x)
        else:
            return data[x]


@task(returns=int)
def compute_maximum(X):
    import numpy as np
    maximum = np.linalg.norm(X[0])
    for i in range(len(X)):
        if np.linalg.norm(X[i]) > maximum:
            maximum = np.linalg.norm(X[i])
    return maximum


def init_random(size, dim, seed):
    lst = []
    for i in range(size):
        if seed is not None:
            random.seed(seed+i)
        lst.append(10000 * random.random(dim))
    return lst


@task(returns=list)
def genFragment(size, dim, seed):
    return init_random(size, dim, seed)


def max_norm(numP, dim, numFrag, seed=None):
    from pycompss.api.api import compss_wait_on

    size = int(numP / numFrag)

    X = [genFragment(size, dim, seed) for _ in range(numFrag)]
    maxs = [compute_maximum(X[i]) for i in range(numFrag)]
    result = mergeReduce(maxFinal, maxs)
    result = compss_wait_on(result)
    return result

def main():
    import sys
    import time

    numP = int(sys.argv[1])
    dim = int(sys.argv[2])
    numFrag = int(sys.argv[3])

    startTime = time.time()
    result = max_norm(numP, dim, numFrag)
    print("The maximum norm is %f" % result)
    print("Elapsed Time {} (s)".format(time.time() - startTime))

if __name__ == "__main__":
    main()
