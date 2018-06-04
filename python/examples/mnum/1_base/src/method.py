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


class MethodP(object):

    def __init__(self, fun, err, prec, nump):
        self.fun = fun
        self.err = err
        self.xn = []
        self.n = 0
        self.rho = 0.0
        self.nump = nump
        self.prec = prec
        self.plotInfo = []

    def solution(self, x1):
        from pycompss.api.api import compss_wait_on
        from decimal import Decimal, getcontext
        print("solution: ", getcontext().prec)
        self.err.criteria(self.rho)
        self.xn = [[Decimal(x)] for x in x1]
        self.fun.fn = [[self.fun.eval(self.xn[i][0], False)] for i in range(self.nump)]
        for i in range(self.nump):
            self.xn[i].append(self.iterate(self.xn[i][self.n], self.xn[(i+1) % self.nump][self.n], self.fun.fn[i][self.n], self.fun.fn[(i+1) % self.nump][self.n]))
            self.fun.fn[i].append(self.fun.eval(self.xn[i][self.n+1], False))

        self.err.en = [[self.err.error(self.xn[i][self.n+1], self.xn[i][self.n])] for i in range(self.nump)]
        self.n += 1

        converge = False
        cnv = []
        while not converge:
            for i in range(self.nump):
                self.xn[i].append(self.iterate(self.xn[i][self.n], self.xn[(i+1) % self.nump][self.n], self.fun.fn[i][self.n], self.fun.fn[(i+1) % self.nump][self.n]))
                self.err.en[i].append(self.err.error(self.xn[i][self.n+1], self.xn[i][self.n]))
                self.err.dnAux.append(self.err.decimals(self.err.en[i][self.n], self.err.en[i][self.n-1]))
                cnv.append(self.err.converge(self.err.dnAux[i]))

            '''Add future object to wait for list for plotting'''
            self.plotInfo.append(self.err.dnAux)
            cnv = compss_wait_on(cnv)

            if True in cnv:
                converge = True
            else:
                converge = False
                for i in range(self.nump):
                    self.fun.fn[i].append(self.fun.eval(self.xn[i][self.n+1], False))
            self.n += 1
            print(self.n, cnv)
            if self.n == 5:
                return [self.xn[i][self.n] for i in range(self.nump)]
            self.err.dnAux = []
            cnv = []

        waitForList = [self.xn[i][len(self.xn[i])-1] for i in range(self.nump)]
        waitForList = compss_wait_on(waitForList)
        return waitForList

    def plot(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from pycompss.api.api import compss_wait_on
        print("plot result list")
        print(self.plotInfo)
        plotDec = []

        for l in self.plotInfo:
            aux = compss_wait_on(l)
            plotDec.append(aux)
            print(aux)

        for i in range(len(plotDec)):
            naux = [i+1 for j in range(len(plotDec[i]))]
            d = [float(dn) for dn in plotDec[i]]
            plt.plot(naux, d, 'ro')
        line = [[] for i in range(self.nump)]
        for i in range(len(plotDec)):
            for j in range(len(plotDec[i])):
                line[j].append(plotDec[i][j])
        for l in line:
            nl = np.arange(1, len(l)+1, 1)
            plt.plot(nl, l)
            print("cl: ", l[len(l)-1]/l[len(l)-2])

        plt.plot([0, self.n], [self.prec, self.prec])
        plt.axis([0, self.n, 0, self.prec+30])
        plt.ylabel("correct digits")
        plt.xlabel("iteration")
        plt.show()
        return


class SMethodP(MethodP):

    from pycompss.api.task import task
    from decimal import Decimal

    def __init__(self, fun, err, prec, nump):
        MethodP.__init__(self, fun, err, prec, nump)
        self.rho = (np.sqrt(5.0)+1)/2

    @task(returns=Decimal, priority=True, isModifier=False)
    def iterate(self, x0, x1, f0, f1):
        from decimal import Decimal, getcontext
        getcontext().prec = self.prec
        print("iterate: ", getcontext().prec)
        xn1 = x0 - ((x0-x1)/(f0-f1)) * f0
        return Decimal(xn1)


class SMethodP2(MethodP):

    from pycompss.api.task import task
    from decimal import Decimal

    def __init__(self, fun, err, prec, nump):
        MethodP.__init__(self, fun, err, prec, nump)
        self.rho = (np.sqrt(5.0)+1)/2

    @task(returns=Decimal)
    def iterate(self, x0, x1, f0, f1):
        from decimal import Decimal, getcontext
        getcontext().prec = self.prec
        print("iterate: ", getcontext().prec)
        xn1 = x0 - ((x0-x1)/(f0-f1)) * f0
        return Decimal(xn1)


class KMethodP(MethodP):

    from pycompss.api.task import task
    from decimal import Decimal

    def __init__(self, fun, err, prec, nump):
        MethodP.__init__(self, fun, err, prec, nump)
        self.rho = 2.0

    @task(returns=Decimal, priority=True)
    def iterate(self, x0, x1, f0, f1):
        from decimal import Decimal, getcontext
        getcontext().prec = self.prec
        aux = 2*x0-x1
        faux = self.fun.evalSeq(aux, False)
        xn1 = x0 - ((aux-x1)/(faux-f1))*f0
        return Decimal(xn1)
