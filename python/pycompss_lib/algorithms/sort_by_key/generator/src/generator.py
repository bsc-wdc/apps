#!/usr/bin/python
# -*- coding: utf-8 -*-
#from pycompss.api.task import task


def generateIntData(numRecords, uniqueKeys, uniqueValues, numPartitions, randomSeed):
    recordsPerPartition = int((numRecords / float(numPartitions)))

    def generatePartition(index):
        import random
        effectiveSeed = int(randomSeed) ** index  # .toString.hashCode
        random.seed(effectiveSeed)
        data = [(random.randint(0, uniqueKeys), random.randint(0, uniqueValues))
                for i in range(recordsPerPartition)]
        return data

    return [generatePartition(i) for i in range(numPartitions)]


def generateStringData(numRecords, uniqueKeys, keyLength, uniqueValues, valueLength, numPartitions, randomSeed, storageLocation, hashFunction):
    import pickle
    ints = generateIntData(
        numRecords, uniqueKeys, uniqueValues, numPartitions, randomSeed)

    data = []
    for i in range(numPartitions):
        data.append(map(lambda (k, v): (paddedString(
            k, keyLength, hashFunction), paddedString(v, valueLength, hashFunction)), ints[i]))

    ff = open(storageLocation, 'w')
    pickle.dump(data, ff)

    return data


def paddedString(i, length, hashFunction):
    fmtString = "{:0>" + str(length) + "d}"
    if hashFunction:
        out = hash(i)
        if len(str(out)) < length:
            return fmtString.format(out)
        else:
            return out
    else:
        return fmtString.format(i)


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


def hashedPartitioner(k, numPartitions, keyfunc=lambda x: x):
    return hash(keyfunc(k)) % numPartitions


def keyfunc(x):
    return x

'''Sorts self, which is assumed to consists of (key, value) pairs'''
#@task(returns=list)


def sortPartition(iterator, ascending=True):
    return sorted(iterator, key=lambda (k, v): keyfunc(k), reverse=not ascending)


def sortByKey(self, ascending=True, numPartitions=None, keyfunc=lambda x: x):
    return map(sortPartition, partitionBy(self, numPartitions, True))


def reduceDict(dic1, dic2):
    for k in dic2:
        if k in dic1:
            dic1[k] += dic2[k]
        else:
            dic1[k] = dic2[k]


if __name__ == "__main__":
    import sys
    #from pycompss.api.api import compss_wait_on
    '''test'''
    print generateIntData(10, 10, 10, 2, 5)
    print generateStringData(10, 10, 5, 10, 5, 2, 8, out_path, True)
