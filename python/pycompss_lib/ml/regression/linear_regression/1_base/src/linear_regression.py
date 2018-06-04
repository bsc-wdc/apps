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

"""
@author: scorella
Linear regression
model: y1=alpha+beta*xi+epsiloni
goal: y = alpha + beta*x
"""

import sys
from pycompss.api.task import task
import math
from numpy import arange
from numpy.random import randint


def list_length(l):
    """ Recursive function to get the size of any list """
    if l:
        if not isinstance(l[0], list):
            return 1 + list_length(l[1:])
        else:
            return list_length(l[0]) + list_length(l[1:])
    return 0


def merge_reduce(f, data):
    """ Apply function cumulatively to the items of data,
        from left to right in binary tree structure, so as to
        reduce the data to a single value.
    :param f: function to apply to reduce data
    :param data: List of items to be reduced
    :return: result of reduce the data to a single value
    """
    from collections import deque
    q = deque(range(len(data)))
    while len(q):
        x = q.popleft()
        if len(q):
            y = q.popleft()
            data[x] = f(data[x], data[y])
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
    result = merge_reduce(reduce_add, [_mean(x, n) for x in X])
    if wait:
        from pycompss.api.api import compss_wait_on
        result = compss_wait_on(result)
    return result


@task(returns=list)
def _norm(X, m):
    return [x-m for x in X]


@task(returns=list)
def _pow(X, p=2):
    return [pow(x, p) for x in X]


@task(returns=float)
def _mul(x, y):
    return x*y


def std(X, m, wait=False):
    xs = [_norm(x, m) for x in X]
    xp = [_pow(x, 2) for x in xs]
    n = list_length(X)
    suma = merge_reduce(reduce_add, [_mean(x, n) for x in xp])
    if wait:
        from pycompss.api.api import compss_wait_on
        suma = compss_wait_on(suma)
    return suma


@task(returns=float)
def op_task(sum_x, sum_y, suma):
    return suma/float(math.sqrt(sum_x*sum_y))


def pearson(X, Y, mx, my):
    from pycompss.api.api import compss_wait_on
    xs = [_norm(x, mx) for x in X]
    ys = [_norm(y, my) for y in Y]
    xxs = [_pow(x, 2) for x in xs]
    yys = [_pow(y, 2) for y in ys]

    xs = compss_wait_on(xs)
    ys = compss_wait_on(ys)

    aux = [list(zip(a, b)) for (a, b) in [(x, y) for (x, y) in zip(xs, ys)]]
    suma = merge_reduce(reduce_add, [merge_reduce(reduce_add, [_mul(a, b) for (a, b) in p]) for p in aux])

    sum_x = merge_reduce(reduce_add, list(map(_add, xxs)))
    sum_y = merge_reduce(reduce_add, list(map(_add, yys)))
    r = op_task(sum_x, sum_y, suma)
    r = compss_wait_on(r)
    return r


def fit(X, Y):
    from pycompss.api.api import compss_wait_on
    mx = mean(X)
    my = mean(Y)
    r = pearson(X, Y, mx, my)
    stdx = std(X, mx)
    stdy = std(Y, mx)
    stdx = compss_wait_on(stdx)
    stdy = compss_wait_on(stdy)
    b = r * (math.sqrt(stdy) / math.sqrt(stdx))

    mx = compss_wait_on(mx)
    my = compss_wait_on(my)

    A = my - b*mx

    def line(x):
        return b*x+A

    return line


def linear_regression(num_points, dim, fragments, plot_result=False):

    points_per_frag = num_points / fragments
    # Example data: data = [[[1,2,3],[4,5,6]], [[1,2,3],[4,5,6]]]
    data = [[list(randint(100, size=points_per_frag)) for _ in range(fragments)] for _ in range(dim)]
    line = fit(data[0], data[1])
    print([line(x) for x in arange(0.0, 100.0, 1.0)])

    if plot_result:
        from pylab import scatter, plot, savefig, show
        datax = [item for sublist in data[0] for item in sublist]
        datay = [item for sublist in data[1] for item in sublist]
        scatter(datax, datay, marker='x')
        plot([line(x) for x in arange(0.0, 100.0, 0.1)], arange(0.0, 100.0, 0.1))
        # show()
        savefig('lrd.png')


if __name__ == "__main__":
    num_points = int(sys.argv[1])
    dim = 2
    fragments = int(sys.argv[2])
    if len(sys.argv) <= 4:
        plot_result = False
    else:
        plot_result = bool(sys.argv[4])

    linear_regression(num_points, dim, fragments, plot_result)
