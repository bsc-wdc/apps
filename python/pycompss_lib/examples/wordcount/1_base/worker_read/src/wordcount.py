#!/usr/bin/python
# -*- coding: utf-8 -*-
'''Wordcount self read'''
import pickle
import time
from pycompss.api.task import task
from pycompss.api.parameter import *

@task(returns=dict)
def wordCount_selfRead(pathFile,start,sizeBlock):
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
def reduce(dic1,dic2):
    for k in dic2:
        if k in dic1:
            dic1[k] += dic2[k]
        else:
            dic1[k] = dic2[k]


if __name__ == "__main__":
    import sys
    import os
    from pycompss.api.api import compss_wait_on
    pathFile = sys.argv[1]
    resultFile = sys.argv[2]
    sizeBlock = int(sys.argv[3])

    print "Start"
    start = time.time()
    data = open(pathFile)
    data.seek(0,2)
    file_size = data.tell()
    ind = 0
    
    result = {}
    while ind < file_size:
        presult = wordCount_selfRead(pathFile, ind, sizeBlock)
        reduce(result, presult)
        ind += sizeBlock
    result = compss_wait_on(result)

    end = time.time()-start
    print "Ellapsed Time"
    print end

    aux = list(result.items())
    ff = open(resultFile,'w')
    pickle.dump(aux,ff)
    ff.close()
