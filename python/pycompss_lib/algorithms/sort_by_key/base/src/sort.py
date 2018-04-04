#!/usr/bin/python
# -*- coding: utf-8 -*-
from pycompss.api.task import task

def chunks(l, n, balanced=False):
    if not balanced or not len(l) % n:
        for i in xrange(0, len(l), n):
            yield l[i:i + n]
    else:
        rest = len(l) % n
        start = 0
        while rest:
            yield l[start: start + n + 1]
            rest -= 1
            start += n + 1
        for i in xrange(start, len(l), n):
            yield l[i:i + n]

def partitionBy(self, numPartitions, hash):
    if not hash:
        return chunks(self, len(self) / numPartitions, True)
    else:
        partitions = [[] for n in range(numPartitions)]
        for s in self:
            partitions[hashedPartitioner(s, numPartitions)].append(s)
        return partitions

def hashedPartitioner(k, numPartitions):
    return hash(keyfunc(k)) % numPartitions

def keyfunc(x):
    return x

'''Sorts self, which is assumed to consists of (key, value) pairs'''
@task(returns=list)
def sortPartition(iterator, ascending=True):
	return sorted(iterator, key=lambda (k, v): keyfunc(k), reverse=not ascending)

def sort_by_key(self, ascending=True, numPartitions=None, keyfunc=lambda x: x):
    return map(sortPartition, partitionBy(self, numPartitions, False))

if __name__ == "__main__":
    import sys
    from pycompss.api.api import compss_wait_on
    
    if len(sys.argv) != 3:
        print >> sys.stderr, "Usage: sort <INPUT_FILE> <OUTPUT_FILE>"
    input_path = sys.argv[1]
    out_path = sys.argv[2]

    text = open(input_path)
    lines = []

    for line in text:
        lines += map(lambda x: (x, 1), line.strip().split(" "))
    text.close()

    '''Spark max(totalCoreCount,2) ##1'''
    defaultParallelism = 4                # 4 workers en cloud
    reducer = int(max(defaultParallelism / 2, 2))

    result = sort_by_key(lines, numPartitions=reducer)

    result = compss_wait_on(result)

    print map(lambda x: x[0],result)
