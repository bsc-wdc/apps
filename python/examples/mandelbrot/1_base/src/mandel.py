#!/usr/bin/python
# -*- coding: utf-8 -*-
from pycompss.api.task import task
from pycompss.api.parameter import *
from numpy import NaN, arange, abs, array


def m(a, i=100):
    z = 0
    for n in range(1, i):
        z = z**2 + a
        if abs(z) > 2:
            return n
    return NaN


@task(returns=list)
def groupTasks(y, X, n):
    Z = [0 for _ in xrange(len(X))]
    for ix, x in enumerate(X):
        Z[ix] = m(x + 1j * y, n)
    return Z

if __name__ == "__main__":
    import sys
    import time
    from pycompss.api.api import compss_wait_on
    X = arange(-2, .5, .01)
    Y = arange(-1.0,  1.0, .01)
    Z = [[] for _ in xrange(len(Y))]
    n = int(sys.argv[1])
    
    st = time.time()
    for iy, y in enumerate(Y):
        Z[iy] = groupTasks(y,X, n)
    Z = compss_wait_on(Z)
    print "Ellapsed time (s): {}".format(time.time()-st)
    
    # Plot Result
    #import matplotlib.pyplot as plt
    #Z = array(Z)
    #plt.imshow(Z, cmap='spectral')
    #plt.show()
    #plt.imsave('Mandelbrot',Z, cmap='spectral')
