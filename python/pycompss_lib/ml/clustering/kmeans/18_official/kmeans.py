#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#  Copyright 2002-2015 Barcelona Supercomputing Center (www.bsc.es)
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
import numpy as np
from pycompss.api.task import task
from pycompss.api.parameter import *


def chunks(l, n, balanced=False):
    if not balanced or not len(l) % n:
        for i in xrange(0, len(l), n):
            yield l[i:i + n]
    else:
        rest = len(l) % n
        start = 0
        while rest:
            yield l[start: start + n + 1]
            rest -= 1
            start += n + 1
        for i in xrange(start, len(l), n):
            yield l[i:i + n]


def mergeReduce(function, data):
    """ Apply function cumulatively to the items of data,
        from left to right in binary tree structure, so as to
        reduce the data to a single value.
    :param function: function to apply to reduce data
    :param data: List of items to be reduced
    :return: result of reduce the data to a single value
    """
    from collections import deque
    dataNew = data[:]
    q = deque(xrange(len(dataNew)))
    while len(q):
        x = q.popleft()
        if len(q):
            y = q.popleft()
            dataNew[x] = function(dataNew[x], dataNew[y])
            q.append(x)
        else:
            return dataNew[x]


def distance2(x, y, w=[1, 1]):
    peso = abs(w[0] - w[1]) + 1
    return np.linalg.norm(x - y) * peso


@task(returns=dict)
def cluster_points_partial(XP, mu, ind):
    dic = {}
    XP = np.array(XP)
    for x in enumerate(XP):
        bestmukey = min([(i[0], distance2(x[1], mu[i[0]]))
                         for i in enumerate(mu)], key=lambda t: t[1])[0]
        if bestmukey not in dic:
            dic[bestmukey] = [x[0] + ind]
        else:
            dic[bestmukey].append(x[0] + ind)
    print "keys", len(dic.keys()), len(mu)
    return dic


@task(returns=dict)
def partial_sum(XP, clusters, ind):
    XP = np.array(XP)
    p = [(i, [(XP[j - ind]) for j in clusters[i]]) for i in clusters]
    dic = {}
    for i, l in p:
        dic[i] = (len(l), np.sum(l, axis=0))
    return dic


@task(returns=dict, priority=True)
def reduceCentersTask(a, b):
    for key in b:
        if key not in a:
            a[key] = b[key]
        else:
            a[key] = (a[key][0] + b[key][0], a[key][1] + b[key][1])
    return a


def has_converged(mu, oldmu, epsilon, iter, maxIterations):
    if oldmu != []:
        if iter < maxIterations:
            aux = [np.linalg.norm(oldmu[i] - mu[i]) for i in range(len(mu))]
            distancia = sum(aux)
            if distancia < epsilon * epsilon:
                return True
            else:
                return False
        else:
            # detencion pq se ha alcanzado el maximo de iteraciones
            return True


def distance(p, X):
    return min([np.linalg.norm(np.array(p) - x) for x in X])


def cost(Y, C):
    return sum([distance(x, C)**2 for x in Y])


def bestMuKey(X, C):
    w = [0 for i in xrange(len(C))]
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x - np.array(C[i[0]])))
                        for i in enumerate(C)], key=lambda t: t[1])[0]
        w[bestmukey] += 1
    return w


def probabilities(X, C, l, phi, n):
    np.random.seed(5)
    p = [(l * distance(x, C)**2) / phi for x in X]
    p /= sum(p)
    idx = np.random.choice(n, l, p=p)
    newC = [X[i][0] for i in idx]
    return newC


def init_parallel(X, k, l, initSteps=2):
    import random
    random.seed(5)
    numFrag = len(X)
    ind = random.randint(0, numFrag - 1)
    XP = X[ind]
    C = random.sample(XP, 1)
    phi = sum([cost(x, C) for x in X])

    for i in range(initSteps):
        '''calculate p'''
        c = [probabilities(x, C, l, phi, len(x)) for x in X]
        C.extend([item for sublist in c for item in sublist])
        '''cost distributed'''
        phi = sum([cost(x, C) for x in X])

    '''pick k centers'''
    w = [bestMuKey(x, C) for x in X]
    bestC = [sum(x) for x in zip(*w)]
    bestC = np.argsort(bestC)
    bestC = bestC[::-1]
    bestC = bestC[:k]
    return [C[b] for b in bestC]


def init_random(X, k):
    np.random.seed(2)
    f = len(X)
    kperf = k / f
    ind = [np.random.randint(0, len(X[i]), kperf) for i in range(f)]
    m = []
    for i in range(len(ind)):
        newmu = [X[i][j] for j in ind[i]]
        m += newmu
    return m, ind[0]


def init(X, k, mode):
    if mode == "kmeans++":
        return init_parallel(X, k, k)
    else:
        return init_random(X, k)


def kmeans(data, k, mu=[], numFrag=-1, maxIterations=10, epsilon=1e-4, initMode='random'):
    """ kmeans: starting with a set of randomly chosen initial centers,
    one repeatedly assigns each imput point to its nearest center, and
    then recomputes the centers given the point assigment. This local
    search called Lloyd's iteration, continues until the solution does
    not change between two consecutive rounds or iteration > maxIterations.
    :param data: data
    :param k: num of centroids
    :param numFrag: num fragments, if -1 data is considered chunked
    :param maxIterations: max iterations
    :param epsilon: error threshold
    :return: list os centroids
    """
    from pycompss.api.api import compss_wait_on
    print len(data)
    # Data is already fragmented
    if numFrag == -1:
        numFrag = len(data)
    else:
        # fragment data
        data = [d for d in chunks(data, len(data) / numFrag)]
    if mu == []:
        mu = init(data, k, initMode)

    oldmu = []
    n = 0
    size = int(len(data) / numFrag)
    while not has_converged(mu, oldmu, epsilon, n, maxIterations):
        oldmu = list(mu)
        clusters = [cluster_points_partial(
            data[f], mu, f * size) for f in range(numFrag)]
        partialResult = [partial_sum(
            data[f], clusters[f], f * size) for f in range(numFrag)]

        mu = mergeReduce(reduceCentersTask, partialResult)
        mu = compss_wait_on(mu)
        mu = [mu[c][1] / mu[c][0] for c in mu]
        n += 1
    return mu, clusters, n
