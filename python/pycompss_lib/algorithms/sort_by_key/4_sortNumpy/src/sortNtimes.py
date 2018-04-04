#!/usr/bin/python
# -*- coding: utf-8 -*-
from pycompss.api.task import task

def keyfunc(x):
        return x


'''Sorts self, which is assumed to consists of (key, value) pairs'''
@task(returns=int)
def sortPartition(path, ascending=True):
        import pickle
        import numpy as np
        f = open(path, 'r')
        data = pickle.load(f)
        f.close()
	import operator
        res = sorted(data.items(), key=operator.itemgetter(1), reverse=not ascending)
        #res = np.sort(iterator,kind='mergesort')
        return len(res)
        #return iter(sorted(iterator, key=lambda (k, v): keyfunc(k), reverse=not ascending))


def sortByKey(XPath, ascending=True, numPartitions=None, keyfunc=lambda x: x):
        from pycompss.api.api import compss_wait_on
        n = map(sortPartition, XPath)
        n = compss_wait_on(n)
        return len(n)
        #return self.partitionBy(numPartitions, hashedPartitioner).mapPartitions(sortPartition, True)


if __name__ == "__main__":
        import sys
        import os
        import time
        #from pycompss.api.api import compss_wait_on
        path = sys.argv[1]
        reducer = int(sys.argv[2])

        X = []
        for file in os.listdir(path):
            X.append(path+'/'+file)
        timeList = []
        for i in range(10):
            startTime = time.time()
	    result = sortByKey(X, numPartitions=reducer)
	    timeList.append(time.time()-startTime)
        print "Ellapsed Time(s)"
        print "min: "+str(min(timeList))
        print "max: "+str(max(timeList))
        print "mean: "+str(sum(timeList)/len(timeList))
        print result
