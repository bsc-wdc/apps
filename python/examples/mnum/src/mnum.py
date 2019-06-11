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
import time
from function import Function, fun6
from error import AError
from method import SMethodP
import numpy as np
from decimal import getcontext, setcontext, ExtendedContext

digits = 8192
# prec = (digits*np.log(10.0) / np.log(2.0) + 1.0)


def mnum(n, prec):
    step = float((2.0-1.5)/n)

    setcontext(ExtendedContext)
    getcontext().prec = prec

    f = Function(fun6, prec)
    e = AError(digits, prec)

    m = SMethodP(f, e, prec, n)
    print("Main: ", getcontext().prec)
    initial_aprox = list(np.arange(1.5, 2.0, step))
    start = time.time()
    r = m.solution(initial_aprox)

    print("Elapsed Total Time {} (s)".format(time.time() - start))
    # m.plot()


if __name__ == "__main__":
    prec = 4096
    n = 16
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        prec = int(sys.argv[2])
    else:
        print("Default Params")

    mnum(n, prec)
