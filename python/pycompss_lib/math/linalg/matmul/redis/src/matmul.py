from pycompss.api.constraint import constraint
from pycompss.api.task import task
from pycompss.api.parameter import *
from block import Block
from storage.storage_object import StorageObject
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
    ret = Block(mb)
    ret.make_persistent()
    return ret

@constraint (ComputingUnits="${ComputingUnits}")
@task(returns = 1)
def multiply(a_object, b_object, c_object, MKLProc):
    a = a_object.get_block()
    b = b_object.get_block()
    c = c_object.get_block()
    os.environ["MKL_NUM_THREADS"]=str(MKLProc)
    c += a * b
    old_id = c_object.getID()
    c_object.delete_persistent()
    c_object.make_persistent(old_id)
    return c_object

def multiply_seq(a_object, b_object, c_object):
    a = a_object.get_block()
    b = b_object.get_block()
    c = c_object.get_block()
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
    if len(args) >= 5:
        check = args[4] == 'true'
    else:
        check = False
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
                C[i][j] = multiply(A[i][k], B[k][j], C[i][j], MKLProc)

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

        for i in range(MSIZE):
            for j in range(MSIZE):
                to_check = createBlock(BSIZE, True, MKLProc)
                to_check = sync(to_check)
                C[i][j] = sync(C[i][j])
                for k in range(MSIZE):
                    multiply_seq(A[i][k], B[k][j], to_check)
                if not np.allclose(C[i][j].get_block(), to_check.get_block()):
                    print('Seq and parallel results differ')
                    print('Seq:')
                    print(to_check.get_block())
                    print('Parallel:')
                    print(C[i][j].get_block())
                    sys.exit(0)
                else:
                    print('This block is OK')
        print('Seq and parallel results are OK')
    print(args)
