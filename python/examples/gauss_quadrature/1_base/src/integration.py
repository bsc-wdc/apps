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
Gauss quadrature integration
"""

import numpy as np
import sys
from scipy.special.orthogonal import p_roots
from pycompss.api.task import task
from pycompss.functions.reduce import mergeReduce



@task(returns=float)
def sumTwoAreas(a, b):
    return a+b


def gaussQuadraturePoints(nIP):
    zi, wi = p_roots(nIP)
    return zi, wi


@task(returns=float)
def computeInterval(a, b, h, i, f, nIP):
    f = lambda x: np.sqrt(x ** 2+8) #BUG, la serializacion no encuentra imports?
    xi = a + i * h
    xii = a + (i + 1) * h
    zi, wi = gaussQuadraturePoints(nIP)
    p = ((xii - xi) * zi + (xii + xi)) / 2.0
    result = (xii - xi) / 2 * sum(wi * f(p))
    return result


def gaussQuadrature(m, nIP, a, b, f):
    from pycompss.api.api import compss_wait_on
    intervals = []
    h = (b-a)/float(m)
    for i in range(m):
        result = computeInterval(a, b, h, i, f, nIP)
        intervals.append(result)
    result = mergeReduce(sumTwoAreas, intervals)
    return compss_wait_on(result)


if __name__ == '__main__':
    #m = 16
    #nIP = 3
    #a = 0
    #b = 1
    m = int(sys.argv[1])
    nIP = int(sys.argv[2])
    a = int(sys.argv[3])
    b = int(sys.argv[4])

    f = lambda x: np.sqrt(x ** 2+8)

    integral = gaussQuadrature(m, nIP, a, b, f)
    real = (3/2.0)+np.log(4)
    relativeError = (integral-real)/real

    print integral, real, relativeError
