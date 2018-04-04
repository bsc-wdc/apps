#!/usr/bin/python
# -*- coding: utf-8 -*-
from pycompss.api.task import task
from pycompss.api.parameter import *


@task(returns=dict, priority=True)
def wordCount_selfRead(pathFile, start, sizeBlock):
    fp = open(pathFile)
    fp.seek(start)
    aux = fp.read(sizeBlock)
    fp.close()
    data = aux.strip().split(" ")
    partialResult = {}
    for entry in data:
        if entry not in partialResult:
            partialResult[entry] = 1
        else:
            partialResult[entry] += 1
    return partialResult


@task(dic1=INOUT)
def reduce(dic1, dic2):
    for k in dic2:
        if k in dic1:
            dic1[k] += dic2[k]
        else:
            dic1[k] = dic2[k]


def mergeReduce(partialResult):
    n = len(partialResult)
    act = [j for j in range(n)]
    while n > 1:
        aux = []
        if n % 2:
            reduce(partialResult[act[len(act) - 2]],
                   partialResult[act[len(act) - 1]])
            act.pop(len(act) - 1)
            n -= 1
        for i in range(0, n, 2):
            reduce(partialResult[act[i]],
                   partialResult[act[i + 1]])
            aux.append(act[i])
        act = aux
        n = len(act)
    partialResult[0] = compss_wait_on(partialResult[0])
    return partialResult[0]


if __name__ == "__main__":
    import time
    import sys
    #import pickle
    from pycompss.api.api import compss_wait_on

    pathFile = sys.argv[1]
    resultFile = sys.argv[2]
    sizeBlock = int(sys.argv[3])

    start = time.time()
    data = open(pathFile)
    data.seek(0, 2)
    file_size = data.tell()
    ind = 0
    partialResult = []
    while ind < file_size:
        partialResult.append(wordCount_selfRead(pathFile, ind, sizeBlock))
        ind += sizeBlock
    result = mergeReduce(partialResult)
    end = time.time() - start
    print "Ellapsed Time"
    print end

    #aux = list(result.items())
    #ff = open(resultFile,'w')
    #pickle.dump(aux,ff)
    #ff.close()
