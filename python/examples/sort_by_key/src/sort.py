#!/usr/bin/python
#
#  Copyright 2002-2019 Barcelona Supercomputing Center (www.bsc.es)
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
        print("Num lines: " + str(len(self)) + ", Num patitions: " + str(num_partitions))
        return chunks(self, int(len(self) / num_partitions), True)
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


def sort_by_key(self, num_partitions=None):
    return list(map(sort_partition, partition_by(self, num_partitions, False)))


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

    result = sort_by_key(lines, num_partitions=reducer)

    result = compss_wait_on(result)

    print([x[0] for x in result])


if __name__ == "__main__":
    main()
