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

from pycompss.api.task import task
from pycompss.api.parameter import *

@task(returns=dict)
def wordCount(data):
    partialResult = {}
    for entry in data:
        for word in entry:
            if word not in partialResult:
                partialResult[word] = 1
            else:
                partialResult[word] += 1
    return partialResult

#@task(dic1=INOUT)
@task(returns=dict)
def reduce(dic1,dic2):
    for k in dic2:
        if k in dic1:
            dic1[k] += dic2[k]
        else:
            dic1[k] = dic2[k]
    return dic1


### MAIN PROGRAM ###

def merge_reduce(function, data):
    import Queue
    q = Queue.Queue()
    for i in data:
        q.put(i)
    while not q.empty():
        x = q.get()
        if not q.empty():
            y = q.get()
            q.put(function(x,y))
        else:
            return x

def mergeReduce(partialResult):
    n = len(partialResult)
    act = [j for j in range(n)]
    while n > 1:
        aux = []
        if n%2:
            reduce(partialResult[act[len(act)-2]], partialResult[act[len(act)-1]])
            act.pop(len(act)-1)
            n -= 1
        for i in range(0,n,2):
            reduce(partialResult[act[i]],partialResult[act[i+1]])
            aux.append(act[i])
        act = aux
        n = len(act)

    partialResult[0] = compss_wait_on(partialResult[0])
    return partialResult[0]

if __name__ == "__main__":
    import sys
    import os
    import pickle
    import time
    from pycompss.api.api import compss_wait_on

    path = sys.argv[1]
    resultFile = sys.argv[2]

    partialResult = []
    
    start = time.time()

    for file in os.listdir(path):
        f = open(path + file)
        data = f.readlines()
        partialResult.append(wordCount(data))


    print("All wordCount tasks submitted")

    #result = mergeReduce(partialResult)
    result = merge_reduce(reduce,partialResult)
    result = compss_wait_on(result)    

    end = time.time()
    print "Ellapsed time: "
    print end-start

    #Save Result
    aux = list(result.items())
    ff = open(resultFile,'w')
    pickle.dump(aux,ff)
    ff.close()

    #open Result
    '''
    f = open(resultFile,'r')
    aux = pickle.load(f)
    print(aux)
    '''

    '''
    f = open('./result_word_count.dat','r')
    aux = pickle.load(f)
    print(aux)
    '''
