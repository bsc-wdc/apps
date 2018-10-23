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

@constraint (ComputingUnits="${ComputingUnits}")
@task(returns=list)
def dot(A,B,transposeResult=False,transposeB=False):
    if transposeB:
        B = np.transpose(B)
    if transposeResult:
        return np.transpose(np.dot(A,B))
    return np.dot(A,B)

@constraint (ComputingUnits="${ComputingUnits}")
@task(returns=list)
def sumList(A):
    B = A[0]
    for i in range(1,len(A)):
        B += A[i]
    return B

@constraint (ComputingUnits="${ComputingUnits}")
@task(returns=list)
def sumList4(A,B,C,D):
    return A + B + C + D

@constraint (ComputingUnits="${ComputingUnits}")
@task(returns=list)
def sumList2(A,B):
    return A + B

def reduceSum(A, amount = 4):
    if len(A) == 1:
        return A[0]
    if len(A) == 2:
        return sumList2(A[0],A[1])
    if len(A) == 4:
        return sumList4(A[0],A[1],A[2],A[3])
    if len(A) < (amount + 1):
        return sumList(A)
    listToReduce = []
    for i in range(0, len(A), amount):
        listToSum = []
        for j in range(i, min(len(A), i + amount)):
            listToSum.append(A[j])
        listToReduce.append(reduceSum(listToSum))
    return reduceSum(listToReduce)

def reduceSumGen(A, amount = 4):
    if len(A) < (amount + 1):
        return sumList(A)
    listToReduce = []
    for i in range(0, len(A), amount):
        listToSum = []
        for j in range(i, min(len(A), i + amount)):
            listToSum.append(A[j])
        listToReduce.append(reduceSumGen(listToSum))
    return reduceSumGen(listToReduce)

def multiplyBlocked(A,B,BSIZE,MKLProc, transposeB = False):
    if transposeB:
        newB=[]
        for i in range(len(B[0])):
            newB.append([])
            for j in range(len(B)):
                newB[i].append(B[j][i])
        B = newB
    C = []
    for i in range(len(A)):
        C.append([])
        for j in range(len(B[0])):
            listToSum = []
            for k in range(len(A[0])):
                listToSum.append(dot(A[i][k], B[k][j], transposeB=transposeB))
            C[i].append(reduceSum(listToSum))
    return C

if __name__ == "__main__":
    import time
    begginingTime = time.time()
    import sys
    from pycompss.api.api import compss_wait_on, compss_barrier

    args = sys.argv[1:]

    MSIZE = int(args[0])
    BSIZE = int(args[1])
    ProcCoreCount = int(args[2])
    MKLProc = int(args[3])
    A = []
    B = []

    startTime = time.time()

    initialize_variables(MKLProc)

    compss_barrier()

    initTime = time.time() - startTime
    startMulTime = time.time()

    C = multiplyBlocked(A,B,BSIZE,MKLProc)

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
