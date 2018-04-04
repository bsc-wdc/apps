#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from pycompss.api.task import task
from pycompss.api.parameter import *

@task(returns=dict)
def cluster_points_partial(XP, mu, ind):
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
    p = [(i, [(XP[j - ind]) for j in clusters[i]]) for i in clusters]
    dic = {}
    for i, l in p:
        dic[i] = (len(l), np.sum(l, axis=0))
    return dic


@task(returns=list)
def reduceCenters(partialResult):
    centers = {}
    for l in partialResult:
        for key in l:
            if key not in centers:
                centers[key] = l[key]
            else:
                centers[key] = (
                    centers[key][0] + l[key][0], centers[key][1] + l[key][1])

    return [centers[c][1] / centers[c][0] for c in centers]


def generate_data(numV, dim, K):
    n = int(float(numV) / K)
    data = []

    for k in range(K):
        c = [random.uniform(-1, 1) for i in range(dim)]
        s = random.uniform(0.05, 0.5)
        for i in range(n):
            d = np.array([np.random.normal(c[j], s) for j in range(dim)])
            data.append(d)

    Data = np.array(data)[:numV]
    return Data


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


def distance(p,X):
    return min([np.linalg.norm(p-x) for x in X])

@task(returns=int)
def cost(Y,C):
    return sum([distance(y,C)**2 for y in Y])

def chunks(l, n, balanced=False):
    if not balanced or not len(l) % n:
        for i in xrange(0, len(l), n):
            yield l[i:i + n]
    else:
        rest = len(l) % n
        start = 0
        while rest:
            yield l[start: start+n+1]
            rest -= 1
            start += n+1
        for i in xrange(start, len(l), n):
            yield l[i:i + n]

def init_random(X,k):
    return random.sample(X,k)

def init_parallel(X,k,initSteps,l,frag):
    from pycompss.api.api import compss_wait_on
    C = random.sample(X,1)
    n = len(X)

    chunkSize = int(n/frag)
    partialX = chunks(X,chunkSize,True)
    psi = [cost(x,C) for x in partialX]
    psi = compss_wait_on(psi)
    phi = sum(psi)

    for i in range(initSteps):
        '''calculate p'''
        p = [(l*distance(x,C)**2)/phi for x in X]
        p /= sum(p)
        '''update C'''
        print(len(p),n)
        idx = np.random.choice(n,l,p=p)
        for i in idx:
            if not any((X[idx] == x).all() for x in C):
                C.append(X[idx][0])
        '''cost distributed'''
        psi = [cost(x,C) for x in partialX]
        psi = compss_wait_on(psi)
        phi = sum(psi)

    '''weight each candidate'''
    w = [0 for i in range(len(C))]
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-C[i[0]])) \
                    for i in enumerate(C)], key=lambda t:t[1])[0]
        w[bestmukey] += 1

    '''pick k centers'''
    bestC = np.argsort(w)
    bestC = bestC[::-1]
    bestC = bestC[:k]
    
    return [C[b] for b in bestC]

def init_seq(X,k):
    '''first center'''
    C = random.sample(X,1)
    print C
    print type(C)
    p = [(distance(x,C)**2)/cost(X,C) for x in X]
    p /= sum(p)
    n = len(X)
    while len(C) < k:
        idx = np.random.choice(n,1,p=p)
        if not any((X[idx] == x).all() for x in C):
            C.append(X[idx][0])
        p = [(distance(x,C)**2)/cost(X,C) for x in X]
        p /=sum(p)
    return C

def init(X,k,mode,numFrag,initIter=5):
    if mode == "kmeans++":
        return init_seq(X,k)
    elif mode == "kmeans++||":
        return init_parallel(X,k,initIter,k,numFrag)
    else:
        return init_random(X,k)

def kmeans_frag(X, k, epsilon, maxIterations, numFrag, initMode):
    import random
    import numpy as np
    from pycompss.api.api import compss_wait_on

    mu = init(X,k,initMode,numFrag)
    oldmu = []
    n = 0
    size = int(len(X) / numFrag)
    while not has_converged(mu, oldmu, epsilon, n, maxIterations):
        oldmu = mu
        clusters = [cluster_points_partial(
            X[f * size:f * size + size], mu, f * size) for f in range(numFrag)]
        partialResult = [partial_sum(
            X[f * size:f * size + size], clusters[f], f * size) for f in range(numFrag)]

        partialResult = compss_wait_on(partialResult)

        mu = reduceCenters(partialResult)
        mu = compss_wait_on(mu)
        print mu
        n += 1
    return n

if __name__ == "__main__":
    import sys
    import random
    import time

    numV = int(sys.argv[1])
    dim = int(sys.argv[2])
    k = int(sys.argv[3])
    numFrag = int(sys.argv[4])
    initMode = sys.argv[5]

    X = generate_data(numV, dim, k)
    startTime = time.time()
    result = kmeans_frag(X, k, 1e-8, 10, numFrag, initMode)
    print "Ellapsed Time(s)"
    print time.time()-startTime

    print result
