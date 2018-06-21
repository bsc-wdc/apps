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

import os
from pycompss.api.task import task
from pycompss.api.parameter import *
import numpy as np
import pickle


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


def init_random(dim, k, seed):
    np.random.seed(seed)
    m = np.random.random((k, dim))
    return m


@task(returns=dict)
def cluster_points_partial(XP, mu, ind):
    """
    For each point computes the nearest center.
    :param XP: Fragments of points
    :param mu: Centers
    :param ind: point first index
    :return: {mu_ind: [pointInd_i, ..., pointInd_n]}
    """
    dic = {}
    for x in enumerate(XP):
        bestmukey = min([(i[0], np.linalg.norm(x[1] - mu[i[0]]))
                         for i in enumerate(mu)], key=lambda t: t[1])[0]
        if bestmukey not in dic:
            dic[bestmukey] = [x[0] + ind]
        else:
            dic[bestmukey].append(x[0] + ind)
    return dic


@task(returns=dict)
def partial_sum(XP, clusters, ind):
    """
    For each cluster returns the number of points and the sum of all the
    points that belong to the cluster.
    :param XP: points
    :param clusters: partial cluster {mu_ind: [pointInd_i, ..., pointInd_n]}
    :param ind: point first ind
    :return: {cluster_ind: (#points, sum(points))}
    """
    dic = {}
    for i in clusters:
        p_idx = np.array(clusters[i]) - ind
        dic[i] = (len(p_idx), np.sum(XP[p_idx], axis=0))
    return dic


@task(returns=dict, priority=True)
def reduceCentersTask(a, b):
    """
    Reduce method to sum the result of two partial_sum methods
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
    """
    Check the convergence
    :param mu: New Centers
    :param oldmu: Previous Centers
    :param epsilon: Convergence distance
    :param iter: Iteration number
    :param maxIterations: Max number of iterations
    :return: True if converged. False on the contrary.
    """
    print "iter: " + str(iter)
    print "maxIterations: " + str(maxIterations)
    if oldmu != []:
        if iter < maxIterations:
            aux = [np.linalg.norm(oldmu[i] - mu[i]) for i in range(len(mu))]
            distancia = sum(aux)
            if distancia < epsilon * epsilon:
                print "Distancia_T: " + str(distancia)
                return True
            else:
                print "Distancia_F: " + str(distancia)
                return False
        else:
            # Maximum number of iterations reached
            return True


@task(returns=list, file_name=FILE_IN)
def readFragment(file_name):
    """
    Reads a file, unpickles its content and retrieves the object
    :param file_name: File to read
    :return: The fragment
    """
    return pickle.load(open(file_name, 'rb'))


def kmeans(dataset_path, numV, k, dim, epsilon, maxIterations):
    """
    Kmeans main code
    :param dataset_path: Dataset path
    :param numV: Number of points
    :param k: Number of centers
    :param dim: Number of dimensions
    :param epsilon: Convergence distance
    :param maxIterations: Max number of iterations
    :return: Tuple(iteration number, centers)
    """
    from pycompss.api.api import compss_wait_on

    seed = 5
    X = [readFragment(dataset_path + os.path.sep + file_name) for file_name in os.listdir(dataset_path)]
    numFrag = len(X)
    size = int(numV / numFrag)

    mu = init_random(dim, k, seed)
    oldmu = []
    n = 0
    startTime = time.time()
    while not has_converged(mu, oldmu, epsilon, n, maxIterations):
        oldmu = mu
        partialResult = []
        for f in xrange(numFrag):
            cluster = cluster_points_partial(X[f], mu, f * size)
            partialResult.append(partial_sum(X[f], cluster, f * size))
        mu = mergeReduce(reduceCentersTask, partialResult)
        mu = compss_wait_on(mu)
        mu = [mu[c][1] / mu[c][0] for c in mu]
        while len(mu) < k:
            indP = np.random.randint(0, size)
            indF = np.random.randint(0, numFrag)
            X[indF] = compss_wait_on(X[indF])
            mu.append(X[indF][indP])
        n += 1
        print("Iteration Time {} (s)".format(time.time() - startTime))
    print("Kmeans Time {} (s)".format(time.time() - startTime))
    return n, mu


if __name__ == "__main__":
    import sys
    import time

    dataset_path = sys.argv[1]
    numV = int(sys.argv[2])
    dim = int(sys.argv[3])
    k = int(sys.argv[4])

    startTime = time.time()
    result = kmeans(dataset_path, numV, k, dim, 1e-4, 10)
    print("Elapsed Time {} (s)".format(time.time() - startTime))
