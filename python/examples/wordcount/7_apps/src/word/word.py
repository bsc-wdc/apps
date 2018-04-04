#
#  Copyright 2002-2015 Barcelona Supercomputing Center (www.bsc.es)
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

from numpy import *
import pickle

import collections
from collections import defaultdict
import time


### TASK SELECTION ###

from pycompss.api.task import task
from pycompss.api.parameter import *

@task(returns=dict)
def wordCount(data):
    partialResult = {}
    for entry in data:
        if entry not in partialResult:
            partialResult[entry] = 1
        else:
            partialResult[entry] = partialResult[entry] + 1
    return partialResult

@task(dic1=INOUT)
def reduce(dic1,dic2):
    for k in dic2:
        if k in dic1:
            dic1[k] = dic1[k] + dic2[k]
        else:
            dic1[k] = dic2[k]

def fancyPrint(words):
    total = 0
    for w,n in words:
        s = "word: %s, %s" %(w,str(n))
        print s
        total = total + n
    print "total: %s" %(str(total))

def test(a):
    total = 0
    for w in a:
        total = total + a[w]
    print "total: %s" %(str(total))

### MAIN PROGRAM ###

if __name__ == "__main__":
    import sys
    from pycompss.api.api import compss_wait_on

    textFile = sys.argv[1]
    resultFile = sys.argv[2]
    text = open(textFile)
    partialResult = []

    start = time.time()

    result = {}
    ind = 0
    t = 0
    for line in text:
        k = line.strip().split(" ")
        partialResult.append(wordCount(k))
        ind = ind + 1
        t = t + len(k)
    print("All tasks submitted")

    #Reduce result
    n = ind
    act = [j for j in range(n)]
    while n > 1:
        aux = []
        if n%2:
            reduce(partialResult[act[len(act)-2]], partialResult[act[len(act)-1]])
            act.pop(len(act)-1)
            n = n-1
        for i in range(0,n,2):
            reduce(partialResult[act[i]],partialResult[act[i+1]])
            aux.append(act[i])
        act = aux
        n = len(act)

    partialResult[0] = compss_wait_on(partialResult[0])


    end = time.time()
    print "Ellapsed time: "
    print end-start

    #Test result
    fancyPrint(partialResult[0].items())
    print t

    #Save Result
    aux = list(partialResult[0].items())
    ff = open(resultFile,'w')
    pickle.dump(aux,ff)
    ff.close()

    #open Result
    '''
    f = open(resultFile,'r')
    aux = pickle.load(f)
    print(aux)
    '''
