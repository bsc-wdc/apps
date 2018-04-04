#!/usr/bin/python
# -*- coding: utf-8 -*-
'''Wordcount Block divide'''
import pickle
import time
from pycompss.api.task import task
from pycompss.api.parameter import *

def read_word(file_object):
    for line in file_object:
        for word in line.split():
            yield word

def read_word_by_word(fp,sizeBlock):
    """Lazy function (generator) to read a file piece by piece in 
    chunks of size approx sizeBlock"""
    data = open(fp)
    block = []
    for word in read_word(data):
        block.append(word)
        if sys.getsizeof(block) > sizeBlock:
            yield block
            block = []
    if block:
        yield block

@task(returns=dict)
def wordCount(data):
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

    start = time.time()
    result={}
    for block in read_word_by_word(pathFile,sizeBlock):
       presult = wordCount(block)  
       reduce(result, presult)
    result=compss_wait_on(result)
    end = time.time()-start
    print "Ellapsed Time"
    print end

    #print result
    aux = list(result.items())
    ff = open(resultFile,'w')
    pickle.dump(aux,ff)
    ff.close()
