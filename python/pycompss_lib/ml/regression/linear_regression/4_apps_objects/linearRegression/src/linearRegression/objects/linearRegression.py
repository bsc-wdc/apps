#!/usr/bin/python
# -*- coding: utf-8 -*-
from pycompss.api.task import task
from pycompss.api.parameter import *
import math
from numpy import arange
from numpy.random import randint
import types
from pylab import scatter, show, plot, savefig, sys


# yi = alpha + beta*xi + epsiloni
# goal: y=alpha + betax

def list_length(l):
    """ Recursive function to get the size of any list """
    if l:
        if not isinstance(l[0], list):
            return 1 + list_length(l[1:])
        else:
            return list_length(l[0]) + list_length(l[1:])
    return 0

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


@task(returns=int)
def _add(x):
    return sum(x)


@task(returns=int)
def reduce_add(x, y):
    return x+y


@task(returns=float)
def _mean(X, n):
    return sum(X)/float(n)


def mean(X, wait=False):
    # chunked data
    n = list_length(X)
    result = mergeReduce(reduce_add, [_mean(x, n) for x in X])
    if wait:
        from pycompss.api.api import compss_wait_on
        result = compss_wait_on(result)
    return result


@task(returns=list)
def _norm(X, m):
    return [x-m for x in X]


@task(returns=list)
def _pow(X, p=2):
    return [pow(x, 2) for x in X]


@task(returns=float)
def _mul(x, y):
    return x*y


def std(X, m, wait=False):
    xs = [_norm(x, m) for x in X]
    xp = [_pow(x, 2) for x in xs]
    n = list_length(X)
    suma = mergeReduce(reduce_add, [_mean(x, n) for x in xp])
    if wait:
        from pycompss.api.api import compss_wait_on
        suma = compss_wait_on(suma)
    return suma


@task(returns=float)
def op_task(sum_x, sum_y, suma):
    return suma/float(math.sqrt(sum_x*sum_y))


@task(returns=float)
def multFrag(a, b):
    p = zip(a, b)
    result = 0
    for (a, b) in p:
        result += a * b
    return result


def pearson(X, Y, mx, my):
    xs = [_norm(x, mx) for x in X]
    ys = [_norm(y, my) for y in Y]
    xxs = [_pow(x, 2) for x in xs]
    yys = [_pow(y, 2) for y in ys]

    # xs = compss_wait_on(xs)
    # ys = compss_wait_on(ys)
    # aux = [zip(a,b) for (a,b) in zip(xs,ys)]
    # suma = mergeReduce(reduce_add, [mergeReduce(reduce_add, [_mul(a, b) for (a, b) in p]) for p in aux])

    suma = mergeReduce(reduce_add, [multFrag(a, b) for (a,b) in zip(xs, ys)])

    sum_x = mergeReduce(reduce_add, map(_add, xxs))
    sum_y = mergeReduce(reduce_add, map(_add, yys))
    r = op_task(sum_x, sum_y, suma)
    return r


#@task(returns=types.LambdaType)
@task(returns=(float, float))
def computeLine(r, stdy, stdx, my, mx):
    b = r * (math.sqrt(stdy) / math.sqrt(stdx))
    A = my - b*mx

    #def line(x):
    #    return b*x-A
    #line = lambda x: b*x-A
    #return line
    #return lambda x: b*x-A
    return b, A


def fit(X, Y):
    from pycompss.api.api import compss_wait_on
    mx = mean(X)
    my = mean(Y)
    r = pearson(X, Y, mx, my)
    stdx = std(X, mx)
    stdy = std(Y, mx)

    line = computeLine(r, stdy, stdx, my, mx)

    line = compss_wait_on(line)
    print line
    return lambda x: line[0]*x+line[1]


@task(returns=list)
def genFragment(pointsPerFrag):
    return list(randint(0,100,size=pointsPerFrag))


def initData(pointsPerFrag, fragments, dim):
    data = [[genFragment(pointsPerFrag) for _ in range(fragments)] for _ in range(dim)]
    return data


if __name__ == "__main__":
    numPoints = int(sys.argv[1])
    dim = 2
    fragments = int(sys.argv[3])
    plotResult = bool(sys.argv[4])

    pointsPerFrag = numPoints/fragments
    #data = [[[1,2,3],[4,5,6]], [[1,2,3],[4,5,6]]]  # Test
    #data = [[list(randint(100, size=pointsPerFrag)) for _ in range(fragments)] for _ in range(dim)]
    data = initData(pointsPerFrag, fragments, dim)
    line = fit(data[0], data[1])
    print [line(x) for x in arange(0.0,100.0,1.0)]

    if plotResult:
        datax = [item for sublist in data[0] for item in sublist]
        datay = [item for sublist in data[1] for item in sublist]
        scatter(datax, datay, marker='x')
        plot([line(x) for x in arange(0.0, 100.0, 0.1)], arange(0.0, 100.0, 0.1))
        show()
        savefig('lrd.png')
