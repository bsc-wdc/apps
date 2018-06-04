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

import sys
import time
from pycompss.api.task import task
from numpy import NaN, arange, abs, array


def m(a, i=100):
    z = 0
    for n in range(1, i):
        z = z**2 + a
        if abs(z) > 2:
            return n
    return NaN


@task(returns=list)
def group_tasks(y, X, n):
    Z = [0 for _ in range(len(X))]
    for ix, x in enumerate(X):
        Z[ix] = m(x + 1j * y, n)
    return Z


def mandelbrot(X, Y, Z, points):
    from pycompss.api.api import compss_wait_on

    st = time.time()
    for iy, y in enumerate(Y):
        Z[iy] = group_tasks(y, X, points)
    Z = compss_wait_on(Z)
    print("Elapsed time (s): {}".format(time.time() - st))

    # Plot Result
    import matplotlib.pyplot as plt
    Z = array(Z)
    plt.imshow(Z, cmap='spectral')
    # plt.show()
    plt.imsave('Mandelbrot.png', Z, cmap='spectral')


if __name__ == "__main__":
    X = arange(-2, .5, .01)
    Y = arange(-1.0,  1.0, .01)
    Z = [[] for _ in range(len(Y))]
    n = int(sys.argv[1])

    mandelbrot(X, Y, Z, n)
