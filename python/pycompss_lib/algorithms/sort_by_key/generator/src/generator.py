#!/usr/bin/python
# -*- coding: utf-8 -*-


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
    ints = generateIntData(numRecords, uniqueKeys, uniqueValues, numPartitions, randomSeed)

    data = []
    for i in range(numPartitions):
        data.append([(paddedString(k_v[0], keyLength, hashFunction), paddedString(k_v[1], valueLength, hashFunction)) for k_v in ints[i]])

    ff = open(storageLocation, 'w')
    pickle.dump(data, ff)


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
        for i in range(0, len(l), n):
            yield l[i:i + n]
    else:
        rest = len(l) % n
        start = 0
        while rest:
            yield l[start: start + n + 1]
            rest -= 1
            start += n + 1
        for i in range(start, len(l), n):
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


def main():
    # Tests
    # ints = generateIntData(10, 10, 10, 2, 5)
    # print ints
    # out_path = "/tmp/out.dataset"
    # generateStringData(10, 10, 5, 10, 5, 2, 8, out_path, False)

    import sys

    numRecords = int(sys.argv[1])
    uniqueKeys = int(sys.argv[2])
    keyLength = int(sys.argv[3])
    uniqueValues = int(sys.argv[4])
    valueLength = int(sys.argv[5])
    numPartitions = int(sys.argv[6])
    randomSeed = int(sys.argv[7])
    storageLocation = sys.argv[8]
    hashFunction = bool(sys.argv[9])

    generateStringData(numRecords,
                       uniqueKeys,
                       keyLength,
                       uniqueValues,
                       valueLength,
                       numPartitions,
                       randomSeed,
                       storageLocation,
                       hashFunction)


if __name__ == "__main__":
    main()
