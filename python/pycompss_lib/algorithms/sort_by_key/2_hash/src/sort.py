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


def partition_by(self, num_partitions, hash):
    if not hash:
        return chunks(self, len(self) / num_partitions, True)
    else:
        partitions = [[] for n in range(num_partitions)]
        for s in self:
            partitions[hashed_partitioner(s, num_partitions)].append(s)
        return partitions


def hashed_partitioner(k, num_partitions):
    return hash(keyfunc(k)) % num_partitions


def keyfunc(x):
    return x


@task(returns=list)
def sort_partition(iterator, ascending=True):
    """Sorts self, which is assumed to consists of (key, value) pairs"""
    return sorted(iterator,
                  key=lambda k_v: keyfunc(k_v[0]),
                  reverse=not ascending)


def sortByKey(self, num_partitions=None):
    return list(map(sort_partition, partition_by(self, num_partitions, True)))


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

    default_parallelism = 4
    reducer = int(max(default_parallelism / 2, 2))

    result = sortByKey(lines, num_partitions=reducer)

    result = compss_wait_on(result)

    print([x[0] for x in result])


if __name__ == "__main__":
    main()
