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


class Function(object):

    from decimal import Decimal
    from pycompss.api.task import task
    from pycompss.api.parameter import IN

    def __init__(self, f, prec, alpha=-1):
        self.alpha = alpha
        self.fun = f
        self.fn = []
        self.n = 0
        self.prec = prec

    @task(returns=Decimal, target_direction=IN)
    def eval(self, x, append=True):
        from decimal import getcontext
        getcontext().prec = self.prec
        if append:
            self.fn.append(self.fun(x))
            self.n += 1
            return self.fn[self.n-1]
        else:
            return self.fun(x)

    def evalSeq(self, x, append=True):
        from decimal import getcontext
        getcontext().prec = self.prec
        if append:
            self.fn.append(self.fun(x))
            self.n += 1
            return self.fn[self.n-1]
        else:
            return self.fun(x)


def fun2(x):
    from decimal import Decimal
    if x > 0.0:
        return Decimal(x*x.ln()-Decimal(1.0))
    else:
        return -1.0


def fun6(x):
    from decimal import Decimal
    return Decimal(exp(-x)+cos(x))


def fun3(x):
    from decimal import Decimal, getcontext
    getcontext().prec = 1024
    getcontext().prec += 2
    if x > 0.0:
        res = Decimal(x*x.ln()-Decimal(1.0))
        getcontext().prec -= 2
        return res
    else:
        res = Decimal(-1.0)
        getcontext().prec -= 2
        return res
        return -1.0


def cos(x):
    """Return the cosine of x as measured in radians.

    print cos(Decimal('0.5'))
    0.8775825618903727161162815826
    print cos(0.5)
    0.87758256189
    print cos(0.5+0j)
    (0.87758256189+0j)
    """
    from decimal import getcontext
    getcontext().prec += 2
    i, lasts, s, fact, num, sign = 0, 0, 1, 1, 1, 1
    while s != lasts:
        lasts = s
        i += 2
        fact *= i * (i-1)
        num *= x * x
        sign *= -1
        s += num / fact * sign
    getcontext().prec -= 2
    return +s


def exp(x):
    """Return e raised to the power of x.  Result type matches input type.

    print exp(Decimal(1))
    2.718281828459045235360287471
    print exp(Decimal(2))
    7.389056098930650227230427461
    print exp(2.0)
    7.38905609893
    print exp(2+0j)
    (7.38905609893+0j)
    """
    from decimal import getcontext
    getcontext().prec += 2
    i, lasts, s, fact, num = 0, 0, 1, 1, 1
    while s != lasts:
        lasts = s
        i += 1
        fact *= i
        num *= x
        s += num / fact
    getcontext().prec -= 2
    return +s
