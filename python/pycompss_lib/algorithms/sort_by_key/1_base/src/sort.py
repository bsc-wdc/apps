#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task import task


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


def hashedPartitioner(k, numPartitions):
    return hash(keyfunc(k)) % numPartitions


def keyfunc(x):
    return x


@task(returns=list)
def sortPartition(iterator, ascending=True):
    """Sorts self, which is assumed to consists of (key, value) pairs"""
    return sorted(iterator,
                  key=lambda k_v: keyfunc(k_v[0]),
                  reverse=not ascending)


def sortByKey(self, numPartitions=None):
    return list(map(sortPartition, partitionBy(self, numPartitions, False)))


def main():
    import sys
    from pycompss.api.api import compss_wait_on

    if len(sys.argv) != 2:
        print("Usage: sort <INPUT_FILE>")
    input_path = sys.argv[1]

    text = open(input_path)
    lines = []

    for line in text:
        lines += [(x, 1) for x in line.strip().split(" ")]
    text.close()

    defaultParallelism = 4
    reducer = int(max(defaultParallelism / 2, 2))

    result = sortByKey(lines, numPartitions=reducer)

    result = compss_wait_on(result)

    print([x[0] for x in result])


if __name__ == "__main__":
    main()
