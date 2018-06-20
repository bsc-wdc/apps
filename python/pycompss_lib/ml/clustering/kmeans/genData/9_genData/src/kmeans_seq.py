#!/usr/bin/python
# -*- coding: utf-8 -*-
import random


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


def init_board_gauss(numV, dim, K):
    n = int(float(numV) / K)
    data = []
    random.seed(5)
    for k in range(K):
        c = [random.uniform(-1, 1) for i in range(dim)]
        s = random.uniform(0.05, 0.5)
        for i in range(n):
            d = np.array([np.random.normal(c[j], s) for j in range(dim)])
            data.append(d)

    Data = np.array(data)[:numV]
    return Data


def init_board_random(numV, dim):
    from numpy import random
    random.seed(5)
    return [random.random(dim) for _ in range(numV)]


def cluster_points_partial(XP, mu, ind):
    import numpy as np
    dic = {}
    XP = np.array(XP)
    for x in enumerate(XP):
        bestmukey = min([(i[0], np.linalg.norm(x[1] - mu[i[0]]))
                         for i in enumerate(mu)], key=lambda t: t[1])[0]
        if bestmukey not in dic:
            dic[bestmukey] = [x[0] + ind]
        else:
            dic[bestmukey].append(x[0] + ind)
    return dic


def partial_sum(XP, clusters, ind):
    import numpy as np
    XP = np.array(XP)
    p = [(i, [(XP[j - ind]) for j in clusters[i]]) for i in clusters]
    dic = {}
    for i, l in p:
        dic[i] = (len(l), np.sum(l, axis=0))
    return dic


def reduceCentersTask(a, b):
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
                print "Distancia_T: " + str(distancia)
                return True
            else:
                print "Distancia_F: " + str(distancia)
                return False
        else:
            # detencion pq se ha alcanzado el maximo de iteraciones
            return True


def init_random(dim, k):
    from numpy import random
    random.seed(5)
    #ind = random.randint(0, len(X) - 1)
    m = np.array([random.random(dim) for _ in range(k)])
    # return random.sample(X[ind], k)
    return m


def genFragment(numv, dim):
    # if mode == "gauss":
    #    return init_board_gauss(numv, dim, k)
    # else:
    return init_board_random(numv, dim)


def kmeans_frag(numV, k, dim, epsilon, maxIterations, numFrag):
    import time
    size = int(numV / numFrag)

    startTime = time.time()
    X = [genFragment(size, dim) for _ in range(numFrag)]
    print "Points generation Time {} (s)".format(time.time() - startTime)

    mu = init_random(dim, k)
    oldmu = []
    n = 0
    startTime = time.time()
    while not has_converged(mu, oldmu, epsilon, n, maxIterations):
        oldmu = mu
        clusters = [cluster_points_partial(
            X[f], mu, f * size) for f in range(numFrag)]
        partialResult = [partial_sum(
            X[f], clusters[f], f * size) for f in range(numFrag)]

        mu = mergeReduce(reduceCentersTask, partialResult)
        mu = [mu[c][1] / mu[c][0] for c in mu]
        n += 1
    print "Kmeans Time {} (s)".format(time.time() - startTime)
    return (n, mu)

if __name__ == "__main__":
    import sys
    import time
    import numpy as np

    numV = int(sys.argv[1])
    dim = int(sys.argv[2])
    k = int(sys.argv[3])
    numFrag = int(sys.argv[4])

    startTime = time.time()
    result = kmeans_frag(numV, k, dim, 1e-4, 10, numFrag)
    print "Ellapsed Time {} (s)".format(time.time() - startTime)
