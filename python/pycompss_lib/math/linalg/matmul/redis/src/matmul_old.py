from pycompss.api.constraint import constraint
from pycompss.api.task import task
from pycompss.api.parameter import *
from block import Block
from storage.storage_object import StorageObject
import numpy as np
import os

def initialize_variables(A, B, MKLProc, MSIZE):
    for matrix in [A, B]:
        for i in range(MSIZE):
            matrix.append([])
            for j in range(MSIZE):
                mb = createBlock(BSIZE, False, MKLProc, MSIZE)
                matrix[i].append(mb)
    for i in range(MSIZE):
        C.append([])
        for j in range(MSIZE):
            mb = createMatrix(BSIZE, True, MKLProc, MSIZE)
            C[i].append(mb)

@constraint (ComputingUnits="${ComputingUnits}")
@task(returns=list)
def createBlock(BSIZE, res, MKLProc, MSIZE = 1):
    os.environ["MKL_NUM_THREADS"]=str(MKLProc)
    to_add = np.zeros((BSIZE, BSIZE)) if res else np.random.random((BSIZE, BSIZE))
    if not res:
      for i in range(BSIZE):
        # Work with stochastic matrices to ensure high precision
        to_add[i] /= (np.sum(to_add[i]) * float(MSIZE))
    mb = np.matrix(to_add, dtype=np.double, copy=False)
    ret = Block(mb)
    ret.make_persistent()
    return ret

def createMatrix(BSIZE, res, MKLProc, MSIZE = 1):
    os.environ["MKL_NUM_THREADS"]=str(MKLProc)
    to_add = np.zeros((BSIZE, BSIZE)) if res else np.random.random((BSIZE, BSIZE)) 
    if not res:
      for i in range(BSIZE):
        # Work with stochastic matrices to ensure high precision
        to_add[i] /= (np.sum(to_add[i]) * float(MSIZE))
    mb = np.matrix(np.array(to_add), dtype=np.double, copy=False)
    return mb

@constraint (ComputingUnits="${ComputingUnits}")
@task(c = INOUT)
def multiply(a_object, b_object, c, MKLProc):
    os.environ["MKL_NUM_THREADS"]=str(MKLProc)
    c += a_object.block * b_object.block

def multiply_seq(a_object, b_object, c):
    a = a_object.get_block()
    b = b_object.get_block()
    c += a * b

@task()
def persist_result(obj):
    to_persist = Block(obj)
    to_persist.make_persistent()

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
    check = len(args) >= 5 and args[4] == 'true'
    A = []
    B = []
    C = []

    startTime = time.time()

    initialize_variables(A, B, MKLProc, MSIZE)

    compss_barrier()

    initTime = time.time() - startTime
    startMulTime = time.time()

    for i in range(MSIZE):
        for j in range(MSIZE):
            for k in range(MSIZE):
                # This loop order maximizes "data diversity"
                ks = (i + k) % MSIZE
                multiply(A[i][ks], B[ks][j], C[i][j], MKLProc)

    for i in range(MSIZE):
        for j in range(MSIZE):
            from pycompss.api.api import compss_delete_object as del_obj
            persist_result(C[i][j])
            if not check:
              del_obj(C[i][j])

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

    if check:
        from pycompss.api.api import compss_wait_on as sync
        for i in range(MSIZE):
            for j in range(MSIZE):
                A[i][j] = sync(A[i][j])
                B[i][j] = sync(B[i][j])
                print('Block %d-%d' % (i, j))
                print(A[i][j])
                print(B[i][j])

        for i in range(MSIZE):
            for j in range(MSIZE):
                to_check = createMatrix(BSIZE, True, MKLProc)
                to_check = sync(to_check)
                C[i][j] = sync(C[i][j])
                for k in range(MSIZE):
                    multiply_seq(A[i][k], B[k][j], to_check)
                C[i][j] = sync(C[i][j])
                to_check = sync(to_check)
                print('Block %d-%d' % (i, j))
                print(C[i][j])
                print("-------------------------------------------------")
                if not np.allclose(C[i][j], to_check):
                    print('Seq and parallel results differ')
                    print('Seq:')
                    print(to_check)
                    print('Parallel:')
                    print(C[i][j])
                    sys.exit(0)
                else:
                    print('This block is OK')
        print('Seq and parallel results are OK')
