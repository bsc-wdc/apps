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

"""
@author: scorella
Gauss quadrature integration
"""

import numpy as np
import sys
from scipy.special.orthogonal import p_roots
from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce


@task(returns=float)
def sum_two_areas(a, b):
    return a+b


def gauss_quadrature_points(nIP):
    zi, wi = p_roots(nIP)
    return zi, wi


@task(returns=float)
def compute_interval(a, b, h, i, f, nIP):
    xi = a + i * h
    xii = a + (i + 1) * h
    zi, wi = gauss_quadrature_points(nIP)
    p = ((xii - xi) * zi + (xii + xi)) / 2.0
    result = (xii - xi) / 2 * sum(wi * np.sqrt(p ** 2 + 8))
    return result


def gauss_quadrature(m, nIP, a, b, f):
    from pycompss.api.api import compss_wait_on
    intervals = []
    h = (b-a)/float(m)
    for i in range(m):
        result = compute_interval(a, b, h, i, f, nIP)
        intervals.append(result)
    result = merge_reduce(sum_two_areas, intervals)
    result = compss_wait_on(result)
    return result


def main(m, nIP, a, b):
    integral = gauss_quadrature(m, nIP, a, b, lambda x: np.sqrt(x ** 2 + 8))
    real = (3 / 2.0) + np.log(4)
    relative_error = (integral - real) / real
    print('Integral      : %d' % integral)
    print('Real          : %d' % real)
    print('Relative error: %d' % relative_error)


if __name__ == '__main__':
    # Example: m = 16
    # Example: nIP = 3
    # Example: a = 0
    # Example: b = 1
    m = int(sys.argv[1])
    nIP = int(sys.argv[2])
    a = int(sys.argv[3])
    b = int(sys.argv[4])

    main(m, nIP, a, b)
