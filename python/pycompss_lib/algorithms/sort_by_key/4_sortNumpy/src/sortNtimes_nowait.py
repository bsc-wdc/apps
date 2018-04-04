#!/usr/bin/python
# -*- coding: utf-8 -*-
from pycompss.api.task import task


'''Sorts data which is assumed to consists of (key, value) pairs'''
@task(returns=int)
def sortPartition(file, ascending=True):
        import pickle
        import operator
        f = open(file, 'r')
        data = pickle.load(f)
        f.close()
        res = sorted(data.items(), key=operator.itemgetter(1), reverse=not ascending)
        return len(res)


def sortByKey(files, ascending=True, numPartitions=None):
        from pycompss.api.api import compss_wait_on
        fo_list = []
        for i in range(10):
            fo_list.append(map(sortPartition, files))    
	result_list = compss_wait_on(fo_list)
        return len(result_list)


if __name__ == "__main__":
        import sys
        import os
        import time
        from pycompss.api.api import compss_wait_on
        path = sys.argv[1]
        reducer = int(sys.argv[2])

        files = []
        for file in os.listdir(path):
            files.append(path+'/'+file)
        
	startTime = time.time()
	result = sortByKey(files, numPartitions=reducer)
	endTime = time.time()-startTime
        print "Ellapsed Time(s)"
        print endTime
        print result
