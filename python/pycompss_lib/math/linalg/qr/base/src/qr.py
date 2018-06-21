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
from pycompss.api.constraint import constraint
from pycompss.api.task import task
from pycompss.api.parameter import *


# For the moment, only works for square matrix. No size verifications.


def setMKLNumThreads(MKLProc):
    import os
    os.environ["MKL_NUM_THREADS"] = str(MKLProc)


# ########################################## #
# ########## BLOCK INITIALIZATION ########## #
# ########################################## #

@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=list)
def createBlock(BSIZE, MKLProc, type='random'):
    setMKLNumThreads(MKLProc)
    if type == 'zeros':
        block = np.matrix(np.zeros((BSIZE, BSIZE)), dtype=np.double, copy=False)
    elif type == 'identity':
        block = np.matrix(np.identity(BSIZE), dtype=np.double, copy=False)
    else:
        block = np.matrix(np.random.random((BSIZE, BSIZE)), dtype=np.double, copy=False)
    return block


def genMatrix(MKLProc):
    A = []
    for i in range(MSIZE):
        A.append([])
        for j in range(MSIZE):
            A[i].append(createBlock(BSIZE, MKLProc, type='random'))
    return A


def genZeros(MSIZE, BSIZE, MKLProc):
    A = []
    for i in range(MSIZE):
        A.append([])
        for j in range(0, MSIZE):
            A[i].append(createBlock(BSIZE, MKLProc, type='zeros'))
    return A


def genIdentity(MSIZE, BSIZE, MKLProc):
    A = []
    for i in range(MSIZE):
        A.append([])
        for j in range(0, i):
            A[i].append(createBlock(BSIZE, MKLProc, type='zeros'))
        A[i].append(createBlock(BSIZE, MKLProc, type='identity'))
        for j in range(i+1, MSIZE):
            A[i].append(createBlock(BSIZE, MKLProc, type='zeros'))
    return A


# ########################################## #
# ########### MATHEMATICAL TASKS ########### #
# ########################################## #

@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=(list, list))
def main_qr(A, MKLProc, mode='reduced', transpose=False):
    from numpy.linalg import qr
    setMKLNumThreads(MKLProc)
    (Q, R) = qr(A, mode=mode)
    if transpose:
        Q = np.transpose(Q)
    return Q, R


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=list)
def dot(A, B, MKLProc, transposeResult=False, transposeB=False):
    setMKLNumThreads(MKLProc)
    if transposeB:
        B = np.transpose(B)
    if transposeResult:
        return np.transpose(np.dot(A, B))
    return np.dot(A, B)


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=(list, list, list, list, list, list))
def littleQR(A, B, MKLProc, BSIZE, transpose=False):
    setMKLNumThreads(MKLProc)
    currA = np.bmat([[A], [B]])
    (subQ, subR) = np.linalg.qr(currA, mode='complete')
    AA = subR[0:BSIZE]
    BB = subR[BSIZE:2 * BSIZE]
    subQ = splitMatrix(subQ, 2)
    if transpose:
        return np.transpose(subQ[0][0]), np.transpose(subQ[1][0]), np.transpose(subQ[0][1]), np.transpose(subQ[1][1]), AA, BB
    else:
        return subQ[0][0], subQ[0][1], subQ[1][0], subQ[1][1], AA, BB


@constraint(ComputingUnits="${ComputingUnits}")
@task(C=INOUT)
def multiplySingleBlock(A, B, C, MKLProc, transposeB=False):
    setMKLNumThreads(MKLProc)
    if transposeB:
        B = np.transpose(B)
    C += A * B


# ########################################## #
# ######### MATHEMATICAL FUNCTIONS ######### #
# ########################################## #

def copyBlocked(A, transpose=False):
    B = []
    for i in range(len(A)):
        B.append([])
        for j in range(len(A[0])):
            B[i].append(np.matrix([0]))
    for i in range(len(A)):
        for j in range(len(A[0])):
            if transpose:
                B[j][i] = A[i][j]
            else:
                B[i][j] = A[i][j]
    return B


def multiplyBlocked(A, B, BSIZE, MKLProc, transposeB=False):
    if transposeB:
        newB = []
        for i in range(len(B[0])):
            newB.append([])
            for j in range(len(B)):
                newB[i].append(B[j][i])
        B = newB
    C = []
    for i in range(len(A)):
        C.append([])
        for j in range(len(B[0])):
            C[i].append(createBlock(BSIZE, MKLProc, type='zeros'))
            for k in range(len(A[0])):
                multiplySingleBlock(A[i][k], B[k][j], C[i][j], MKLProc, transposeB=transposeB)
    return C


def qr_blocked(A, MKLProc, overwrite_a=False):

    Q = genIdentity(MSIZE, BSIZE, MKLProc)

    if not overwrite_a:
        R = copyBlocked(A)
    else:
        R = A

    for i in range(MSIZE):

        actQ, R[i][i] = main_qr(R[i][i], MKLProc, transpose=True)

        for j in range(MSIZE):
            Q[j][i] = dot(Q[j][i], actQ, MKLProc, transposeB=True)
        for j in range(i+1, MSIZE):
            R[i][j] = dot(actQ, R[i][j], MKLProc)

        # Update values of the respective column
        for j in range(i+1, MSIZE):
            subQ = [[np.matrix(np.array([0])), np.matrix(np.array([0]))],
                    [np.matrix(np.array([0])), np.matrix(np.array([0]))]]
            subQ[0][0], subQ[0][1], subQ[1][0], subQ[1][1], R[i][i], R[j][i] = littleQR(R[i][i],
                                                                                        R[j][i],
                                                                                        MKLProc,
                                                                                        BSIZE,
                                                                                        transpose=True)
            # Update values of the row for the value updated in the column
            for k in range(i + 1, MSIZE):
                [[R[i][k]], [R[j][k]]] = multiplyBlocked(subQ, [[R[i][k]], [R[j][k]]], BSIZE, MKLProc)
            for k in range(MSIZE):
                [[Q[k][i], Q[k][j]]] = multiplyBlocked([[Q[k][i], Q[k][j]]], subQ, BSIZE, MKLProc, transposeB=True)
    return Q, R


def insertMatrix(i, j, auxQ, subQ):
    # Useful to generate the Givens' rotation 'identity' matrix
    auxQ[i][i] = subQ[0][0]
    auxQ[i][j] = subQ[0][1]
    auxQ[j][i] = subQ[1][0]
    auxQ[j][j] = subQ[1][1]


# ########################################## #
# ####### FUNCTIONS TO HANDLE BLOCKS ####### #
# ########################################## #

def joinMatrix(A):
    joinMat = np.matrix([[]])
    for i in range(0, len(A)):
        currRow = A[i][0]
        for j in range(1, len(A[i])):
            currRow = np.bmat([[currRow, A[i][j]]])
        if i == 0:
            joinMat = currRow
        else:
            joinMat = np.bmat([[joinMat], [currRow]])
    return np.matrix(joinMat)


def splitMatrix(A, MSIZE):
    splittedMatrix = []
    bSize = len(A) / MSIZE
    for i in range(MSIZE):
        splittedMatrix.append([])
        for j in range(MSIZE):
            block = []
            for k in range(bSize):
                block.append([])
                for w in range(bSize):
                    block[k].append(A[i * bSize + k, j * bSize + w])
            splittedMatrix[i].append(np.matrix(block))
    return splittedMatrix


# ########################################## #
# ################## MAIN ################## #
# ########################################## #

def qr():
    import time
    import sys
    from pycompss.api.api import compss_barrier
    from pycompss.api.api import compss_wait_on

    np.set_printoptions(precision=2)

    global MSIZE
    global BSIZE
    global mkl_threads

    MSIZE = int(sys.argv[1])
    BSIZE = int(sys.argv[2])
    mkl_threads = int(sys.argv[3])
    verifyOutput = sys.argv[4] == "False"

    startTime = time.time()

    # Generate de matrix
    m2b = genMatrix(mkl_threads)

    compss_barrier()

    initTime = time.time() - startTime

    startDecompTime = time.time()

    (Q, R) = qr_blocked(m2b, mkl_threads)

    compss_barrier()

    decompTime = time.time() - startDecompTime
    totalTime = time.time() - startTime

    print("PARAMS:------------------")
    print("MSIZE:{}".format(MSIZE))
    print("BSIZE:{}".format(BSIZE))
    print("initT:{}".format(initTime))
    print("decompT:{}".format(decompTime))
    print("totalTime:{}".format(totalTime))

    if(verifyOutput):
        Q = compss_wait_on(Q)
        R = compss_wait_on(R)
        m2b = compss_wait_on(m2b)
        print("Input matrix")
        print(joinMatrix(m2b))
        print("Q*R")
        print(joinMatrix(Q)*joinMatrix(R))
        print("Generated R")
        print(joinMatrix(R))
        print("NumPy R")
        print(np.linalg.qr(joinMatrix(m2b))[1])
        print("Generated Q")
        print(joinMatrix(Q))
        print("NumPy Q")
        print(np.linalg.qr(joinMatrix(m2b))[0])


if __name__ == "__main__":
    qr()
