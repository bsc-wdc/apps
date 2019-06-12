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

import sys
import numpy as np
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on


def fft(a):
    x = a.flatten()
    n = x.size

    # precompute twiddle factors
    w = np.zeros((n, n), dtype=complex)

    for i in range(n):
        for j in range(n):
            w[i, j] = np.exp(-2 * np.pi * 1j * j / (i + 1))

    lin = []

    for xk in x:
        lin.append(np.array(xk, ndmin=1, dtype=complex))

    while len(lin) > 1:
        lout = []
        ln = len(lin)

        for k in range(int(ln / 2)):
            lout.append(reduce(lin[k], lin[k + int(ln / 2)], w))

        lin = lout

    return lin[0]


@task(returns=1)
def reduce(even, odd, w):
    x = np.concatenate((even, odd))
    n = len(x)

    for k in range(int(n / 2)):
        e = x[k]
        o = x[k + int(n / 2)]
        wk = w[n - 1, k]

        x[k] = e + wk * o
        x[k + int(n / 2)] = e - wk * o

    return x


if __name__ == "__main__":
    arr_length = int(sys.argv[1])
    arr = np.random.rand(arr_length)
    result = fft(arr)
    result = compss_wait_on(result)
    print("Result: " + str(result))
