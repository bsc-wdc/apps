from pycompss.api.constraint import constraint
from pycompss.api.task import task
from pycompss.api.parameter import *
import numpy as np
import ctypes
import os

## UNCOMMENT IN CASE MKL IS INSTALLED IN YOUR MACHINE

#try:
#    mkl_rt = ctypes.CDLL('libmkl_rt.so')
#except:
#    use_mkl = False

def mkl_set_num_threads(cores):
#    if use_mkl:
#        mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))
    pass

def initialize_variables(MSIZE, BSIZE, MKLProc):
    A = []
    B = []
    C = []
    for matrix in [A, B]:
        for i in range(MSIZE):
            matrix.append([])
            for j in range(MSIZE):
                mb = createBlock(BSIZE, False, MKLProc)
                matrix[i].append(mb)
    for i in range(MSIZE):
        C.append([])
        for j in range(MSIZE):
            mb = createBlock(BSIZE, True, MKLProc)
            C[i].append(mb)
    return A, B, C

@constraint (ComputingUnits="${ComputingUnits}")
@task(returns=list)
def createBlock(BSIZE, res, MKLProc):
    mkl_set_num_threads(MKLProc)
    if res:
        block = np.array(np.zeros((BSIZE, BSIZE)), dtype=np.double, copy=False)
    else:
        block = np.array(np.random.random((BSIZE, BSIZE)), dtype=np.double,copy=False)
    mb = np.matrix(block, dtype=np.double, copy=False)
    return mb

@constraint (ComputingUnits="${ComputingUnits}")
@task(c=INOUT)
def multiply(a, b, c, MKLProc):
    mkl_set_num_threads(MKLProc)
    c += a * b

def dot(A, B, C, MSIZE, MKLProc):
    for i in range(MSIZE):
        for j in range(MSIZE):
            for k in range(MSIZE):
                multiply(A[i][k], B[k][j], C[i][j], MKLProc)

def main():
    import time, sys
    from pycompss.api.api import barrier
    
    args = sys.argv[1:]

    MSIZE = int(args[0])
    BSIZE = int(args[1])
    MKLProc = int(args[2]) 
    
    print(MSIZE)
    print(BSIZE)
    print(MKLProc)

    start_time = time.time()

    A, B, C = initialize_variables(MSIZE, BSIZE, MKLProc)

    barrier()

    start_mult_time = time.time()
    init_time = start_mult_time - start_time

    dot(A, B, C, MSIZE, MKLProc)

    barrier()

    mul_time = time.time() - start_mult_time
    total_time = time.time() - start_time

    print("PARAMS:------------------")
    print("MSIZE: " + str(MSIZE))
    print("BSIZE: " + str(BSIZE))
    print("MKL threads: " + str(MKLProc))
    print("Initialization time: " + str(start_mult_time))
    print("Multiplication time: " + str(mul_time))
    print("Total time: " + str(total_time))

if __name__ == "__main__":
    main()

