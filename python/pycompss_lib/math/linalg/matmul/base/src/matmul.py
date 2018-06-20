from pycompss.api.constraint import constraint
from pycompss.api.task import task
from pycompss.api.parameter import *
import numpy as np
import ctypes
import os

mkl_rt = ctypes.CDLL('libmkl_rt.so')

def mkl_set_num_threads(cores):
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))

def initialize_variables(MKLProc):
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

if __name__ == "__main__":
    import time
    begginingTime = time.time()
    import sys
    from pycompss.api.api import barrier
    
    args = sys.argv[1:]
    
    MSIZE = int(args[0])
    BSIZE = int(args[1])
    MKLProc = int(args[2])
    A = []
    B = []
    C = []

    startTime = time.time()

    initialize_variables(MKLProc)

    barrier()

    initTime = time.time() - startTime
    startMulTime = time.time()

    for i in range(MSIZE):
        for j in range(MSIZE):
            for k in range(MSIZE):
                multiply(A[i][k], B[k][j], C[i][j], MKLProc)

    barrier()

    mulTime = time.time() - startMulTime
    mulTransTime = time.time() - startMulTime
    totalTime = time.time() - startTime
    totalTimeWithImports = time.time() - begginingTime
    print("PARAMS:------------------")
    print("MSIZE:{}" + str(MSIZE))
    print("BSIZE:{}" + str(BSIZE))
    print("initT:{}" + str(initTime))
    print("multT:{}" + str(mulTime)) 
    print("mulTransT:{}" + str(mulTransTime))
    print("totalTime:{}" + str(totalTime))
    
