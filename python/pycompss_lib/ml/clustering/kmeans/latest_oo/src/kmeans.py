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

from KMeans import Board
from pycompss.api.task import task
import numpy as np

__author__ = 'Alex Barcelo <alex.barcelo@bsc.es>'
__copyright__ = '2016 Barcelona Supercomputing Center (BSC-CNS)'


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


@task(returns=dict, priority=True)
def reduceCentersTask(a, b):
    """  Reduce method to sum the result of two partial_sum methods
    :param a: partial_sum {cluster_ind: (#points_a, sum(points_a))}
    :param b: partial_sum {cluster_ind: (#points_b, sum(points_b))}
    :return: {cluster_ind: (#points_a+#points_b, sum(points_a+points_b))}
    """
    for key in b:
        if key not in a:
            a[key] = b[key]
        else:
            a[key] = (a[key][0] + b[key][0], a[key][1] + b[key][1])
    return a


def has_converged(mu, oldmu, epsilon, iter, maxIterations):
    print "iter: " + str(iter)
    print "maxIterations: " + str(maxIterations)
    if oldmu != []:
        if iter < maxIterations:
            aux = [np.linalg.norm(oldmu[i] - mu[i]) for i in range(len(mu))]
            distancia = sum(aux)
            if distancia < epsilon * epsilon:
                print("Distancia_T: " + str(distancia))
                return True
            else:
                print("Distancia_F: " + str(distancia))
                return False
        else:
            return True


def init_random(dim, k, seed):
    np.random.seed(seed)
    m = np.random.random((k, dim))
    return m


def kmeans(numV, k, dim, epsilon, maxIterations, numFrag):
    from pycompss.api.api import compss_wait_on
    size = numV // numFrag  # points per fragment, I assume, and I hope that the division is exact

    startTime = time.time()
    X = Board(dim, size)

    # Initialize locally, make all the fragments persistents, and then persist the board itself
    X.init_random(numV, 5)  # starting seed to match PyCOMPSs original generation
    # X.make_persistent()

    mu = list(init_random(dim, k, 5))  # consistency, mu is a list

    print("Points generation Time {} (s)".format(time.time() - startTime))

    oldmu = []
    n = 0
    startTime = time.time()
    while not has_converged(mu, oldmu, epsilon, n, maxIterations):
        oldmu = mu
        partialResult = list()
        for f, frag in enumerate(X.fragments):
            cluster = frag.cluster_points(mu)
            partialResult.append(frag.partial_sum(cluster))

        mu = mergeReduce(reduceCentersTask, partialResult)
        mu = compss_wait_on(mu)
        mu = [mu[c][1] / mu[c][0] for c in mu]
        while len(mu) < k:
            indP = np.random.randint(0, size)
            indF = np.random.randint(0, numFrag)
            # not-so-subtle bug here for dataClay implementation of Board --fixme maybe?
            mu.append(X[indF][indP])
        n += 1

    print("Kmeans Time {} (s)".format(time.time() - startTime))
    return n, mu


if __name__ == "__main__":
    import sys
    import time

    numV = int(sys.argv[1])
    dim = int(sys.argv[2])
    k = int(sys.argv[3])
    numFrag = int(sys.argv[4])

    startTime = time.time()
    result = kmeans(numV, k, dim, 1e-4, 10, numFrag)
    print("Elapsed Time {} (s)".format(time.time() - startTime))
