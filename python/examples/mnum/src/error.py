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

import numpy as np


class Error(object):

    def __init__(self, digits, precision):
        self.en = []
        self.dn = 0.0
        self.dnAux = []
        self.p = 0.0
        self.n = 0
        self.digits = digits
        self.precision = precision


class AError(Error):

    from pycompss.api.task import task
    from pycompss.api.parameter import IN
    from decimal import Decimal

    @task(returns=Decimal)
    def error(self, xn1, x1, append=True):
        from decimal import Decimal, getcontext
        getcontext().prec = self.precision
        if append:
            self.en.append(Decimal(xn1-x1))
            self.n += 1
            return self.en[self.n-1]
        else:
            return Decimal(xn1-x1)

    @task(returns=Decimal, target_direction=IN)
    def decimals(self, e1, e0):
        return -self.p*(abs(e1/e0)).log10()

    def criteria(self, rho):
        from decimal import Decimal
        rho = Decimal(rho)
        self.p = Decimal((rho*rho) / (rho-Decimal(1.0)))

    @task(returns=int, target_direction=IN)
    def converge(self, d):
        return d > self.digits


class CError(Error):

    def __init__(self, digits, precision, alpha):
        Error.__init__(self, digits, precision)
        self.alpha = alpha

    def error(self, xn1, append=True):
        if append:
            self.en.append(xn1-self.alpha)
            self.n += 1
            return self.en[self.n-1]
        else:
            return xn1-self.alpha

    def decimals(self):
        self.dn = -np.log10(abs(self.en[self.n-1]))

    def criteria(self, rho):
        self.p = rho

    def converge(self):
        return self.dn > self.digits
