# -*- coding: utf-8 -*-
from __future__ import print_function

#def ones(shape, dtype=None, order='C'):
#    pass

def printOnes():
    print("Print ones")

class array(object):
    def __init__(self, data, dtype=None, copy=True, order=None, subok=False, ndmin=0):
        from matlib import arrayDist
        from numpy import array as numpy_array
        self.__numpy_array__ = numpy_array(data, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)
        self.__name__ = "matlib.core.numeric.array"

    def __getattr__(self, name, mode="Normal"):
        if mode == "Normal":
            return self.__getattr__(name)
        print("#######################################################################################################")

    def __getattr__(self, name):
        return self.__numpy_array__.__getattribute__(name)

    def __getattribute__(self,name):
        return object.__getattribute__(self,name)

    def __str__(self):
        return self.__numpy_array__.__str__()

    def __mul__(self,other):
        from matlib import multiply
        return multiply(self,other)

    def __rmul__(self,other):
        from matlib import multiply
        return multiply(self,other)

    def __add__(self, other):
        from matlib import add
        return add(self,other)

"""
class ndarray(object):
    def __init__(self, shape, dtype=None, buffer=None, offset=0, strides=None, order=None):
        from numpy import ndarray as numpy_ndarray
        self.__numpy_ndarray__ = numpy_ndarray(shape, dtype=dtype, buffer=buffer, offset=offset, strides=strides, order=order)
        self.__name__ = "matlib.core.numeric.ndarray"

    def __str__(self):
        return self.__numpy_ndarray__.__str__()

    def __getattr__(self,name):
        return self.__numpy_ndarray__.__getattribute__(name)

    def __getattribute__(self,name):
        return object.__getattribute__(self,name) 
"""