from pycompss.api.constraint import constraint
from pycompss.api.task import task
from pycompss.api.parameter import *
import numpy as np
import os

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
    os.environ["MKL_NUM_THREADS"]=str(MKLProc)
    if res:
        block = np.array(np.zeros((BSIZE, BSIZE)), dtype=np.double, copy=False)
    else:
        block = np.array(np.random.random((BSIZE, BSIZE)), dtype=np.double,copy=False)
    mb = np.matrix(block, dtype=np.double, copy=False)
    return mb

@constraint (ComputingUnits="${ComputingUnits}")
@task(c=INOUT)
def multiply(a, b, c, MKLProc):
    os.environ["MKL_NUM_THREADS"]=str(MKLProc)
    c += a * b

if __name__ == "__main__":
    import time
    begginingTime = time.time()
    import sys
    from pycompss.api.api import compss_barrier

    args = sys.argv[1:]

    MSIZE = int(args[0])
    BSIZE = int(args[1])
    ProcCoreCount = int(args[2])
    MKLProc = int(args[3])
    A = []
    B = []
    C = []

    startTime = time.time()

    initialize_variables(MKLProc)

    compss_barrier()

    initTime = time.time() - startTime
    startMulTime = time.time()

    for i in range(MSIZE):
        for j in range(MSIZE):
            for k in range(MSIZE):
                multiply(A[i][k], B[k][j], C[i][j], MKLProc)

    compss_barrier()

    mulTime = time.time() - startMulTime
    mulTransTime = time.time() - startMulTime
    totalTime = time.time() - startTime
    totalTimeWithImports = time.time() - begginingTime
    print "PARAMS:------------------"
    print "MSIZE:{}".format(MSIZE)
    print "BSIZE:{}".format(BSIZE)
    print "initT:{}".format(initTime)
    print "multT:{}".format(mulTime)
    print "mulTransT:{}".format(mulTransTime)
    print "procCore:{}".format(ProcCoreCount)
    print "totalTime:{}".format(totalTime)
