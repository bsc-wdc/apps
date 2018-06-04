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

'''Wordcount self read'''
import pickle
import time
from pycompss.api.task import task
from pycompss.api.parameter import *

@task(returns=dict,priority=True)
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
            partialResult[entry] = partialResult[entry] + 1
    return partialResult

@task(returns=dict)
def reduce(dic1,dic2):
    for k in dic2:
        if k in dic1:
            dic1[k] = dic1[k] + dic2[k]
        else:
            dic1[k] = dic2[k]
    return dic1

def mergeReduce(partialResult):
    n = len(partialResult)
    act = [j for j in range(n)]
    while n > 1:
        aux = []
        if n%2:
            partialResult[act[len(act)-2]]=reduce(partialResult[act[len(act)-2]], partialResult[act[len(act)-1]])
            act.pop(len(act)-1)
            n = n-1
        for i in range(0,n,2):
            partialResult[act[i]]=reduce(partialResult[act[i]],partialResult[act[i+1]])
            aux.append(act[i])
        act = aux
        n = len(act)

    partialResult[0] = compss_wait_on(partialResult[0])
    return partialResult[0]

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
    partialResult = []
    while ind < file_size:
        partialResult.append(wordCount_selfRead(pathFile, ind, sizeBlock))
        ind += sizeBlock
    result = mergeReduce( partialResult)
    end = time.time()-start
    print "Ellapsed Time"
    print end

    #aux = list(result.items())
    #ff = open(resultFile,'w')
    #pickle.dump(aux,ff)
    #ff.close()

