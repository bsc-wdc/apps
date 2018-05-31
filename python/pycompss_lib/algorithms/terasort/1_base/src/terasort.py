#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#  Copyright 2002-2017 Barcelona Supercomputing Center (www.bsc.es)
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
from pycompss.api.task import task
from pycompss.api.parameter import *
import sys
range_min = 0
range_max = sys.maxint


@task(returns=list)
def genFragment(numEntries, seed):
    """
    Generate a fragment with random numbers.
    A fragment is a list of tuples, where the first element of each tuple is the key,
    and the second the value.
    Each key is generated randomly between min and max global values.
    Each value is generated randomly between -1 and 1

    fragment structure = [(k1, v1), (k2, v2), ..., (kn, vn)]

    :param numEntries: Number of k,v pairs within a fragment
    :param seed: The seed for the random generator
    :return:
    """
    import random
    random.seed(seed)
    fragment = []     # could be used a python ordered dict?
    for n in range(numEntries):
        fragment.append((random.randrange(range_min, range_max), random.random()))
    return fragment


#@task(returns="tuple([tuple() for i in range(len(fragment))])")
@task(returns=tuple([tuple() for i in range(10)]))  # Multireturn
def filterFragment(fragment, ranges):
    """
    Task that filters a fragment entries for the given ranges.
        * Ranges is a list of tuples where each tuple corresponds to a range.
        * Each tuple (range) is composed by two elements, the minimum and maximum of each range.
        * The filtering is performed by checking which fragment entries' keys belong to each range.
    The entries that belong to each range are considered a bucket.
        * The variable buckets is a list of lists, where the inner lists correspond to the bucket of each range.

    :param fragment: The fragment to be sorted and filtered.
    :param ranges: The ranges to apply when filtering.
    :return: Multireturn of the buckets.
    """
    buckets = []
    for range in ranges:
        buckets.append(filter(lambda (k, v): k >= range[0] and k < range[1], fragment))
    return tuple(buckets)


@task(returns=dict)
def combineAndSortBucketElements(*args):
    """
    Task that combines the buckets received as *args parameter and final sorting.

    *args structure = ([],[], ..., [])

    :param args: *args that contains the buckets of a single range
    :return: A list of tuples with the same format as provided initially sorted by key.
    """
    combined = []
    for e in args:
        for kv in e:
            combined.append(kv)
    sortedByKey = sorted(combined, key=lambda key: key[0])
    return sortedByKey


def generateRanges():
    """
    Generate the ranges of each bucket.
    The ranges structure:
        [(0, a), (a, b), ..., (m, n)]
        * Where (a - 0) == (b - a) == ...
    :return: Ranges of the buckets
    """
    import numpy as np
    split_indexes=np.linspace(range_min, range_max+1,numBuckets+1)
    ranges = []
    for ind in range(split_indexes.size - 1):
        ranges.append((split_indexes[ind], split_indexes[ind + 1]))
    return ranges


def terasort(numFragments, numEntries, numBuckets, seed):
    """
    ----------------------
    Terasort main program
    ----------------------
    This application generates a set of fragments that contain randomly generated key, value tuples
    and sorts them all considering the key of each tuple.

    :param numFragments: Number of fragments to generate
    :param numEntries: Number of entries (k,v tuples) within each fragment
    :param numBuckets: Number of buckets to consider.
    :param seed: Initial seed for the random number generator.
    """
    from pycompss.api.api import compss_wait_on, compss_barrier

    dataset = [genFragment(numEntries, seed + i) for i in range(numFragments)]

    # Init buckets dictionary
    buckets = {}
    for i in range(numBuckets):
        buckets[i] = []

    # Init ranges
    ranges = generateRanges()
    assert(len(ranges) == numBuckets)

    for d in dataset:
        fragmentBuckets = filterFragment(d, ranges) # en fragment
        for i in range(numBuckets):
            buckets[i].append(fragmentBuckets[i])

    # Verify that the buckets contain elements from their corresponding range
    # This verification can not be done when using PyCOMPSs due to buckets contains
    # future objects and would require a synchronization.
    # for i in range(numBuckets):
    #     bucket = buckets[i]
    #     for elem in bucket:
    #         for kv in elem:
    #             assert(kv[0] >= ranges[i][0] and kv[0] < ranges[i][1])

    result = {}
    for key, value in buckets.iteritems():
        result[key] = combineAndSortBucketElements(*tuple(value))  # en bucket

    for key, value in result.iteritems():
        result[key] = compss_wait_on(value)

    print "*********** FINAL RESULT ************"
    import pprint
    pprint.pprint(result)
    print "*************************************"


if __name__ == "__main__":
    import sys
    import time

    arg1 = sys.argv[1] if len(sys.argv) > 1 else 16
    arg2 = sys.argv[2] if len(sys.argv) > 2 else 50

    numFragments = int(arg1) # Default: 16
    numEntries = int(arg2)   # Default: 50
    numBuckets = 10           # int(sys.argv[3]) # be very careful with this argument (since it is in a decorator)
    seed = 5

    startTime = time.time()
    terasort(numFragments, numEntries, numBuckets, seed)
    print "Elapsed Time {} (s)".format(time.time() - startTime)
