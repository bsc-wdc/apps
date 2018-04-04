#!/usr/bin/python
# -*- coding: utf-8 -*-
from pycompss.api.task import task

def keyfunc(x):
        return x


'''Sorts self, which is assumed to consists of (key, value) pairs'''
@task(returns=int)
def sortPartition(iterator, ascending=True):
        import pickle
        import numpy as np
        f = open(iterator, 'r')
        iterator = pickle.load(f)
        f.close()
	#import operator
        #res = sorted(iterator.items(), key=operator.itemgetter(1), reverse=not ascending)
        res = np.sort(iterator,kind='mergesort')
        return len(res)
        #return iter(sorted(iterator, key=lambda (k, v): keyfunc(k), reverse=not ascending))


def sortByKey(self, ascending=True, numPartitions=None, keyfunc=lambda x: x):
        from pycompss.api.api import compss_wait_on
        n = map(sortPartition, self)
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
	startTime = time.time()
        result = sortByKey(X, numPartitions=reducer)
        #result = compss_wait_on(result)
        print "Ellapsed Time(s)"
        print time.time()-startTime
        print result
