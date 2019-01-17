#!/usr/bin/python
#
#  Copyright 2002-2018 Barcelona Supercomputing Center (www.bsc.es)
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
from base.src.tasks import *
import sys

range_min = 0
range_max = sys.maxsize


def generate_ranges(num_buckets):
    """
    Generate the ranges of each bucket.
    The ranges structure:
        [(0, a), (a, b), ..., (m, n)]
        * Where (a - 0) == (b - a) == ...
    :return: Ranges of the buckets
    """
    import numpy as np
    split_indexes = np.linspace(range_min, range_max+1, num_buckets + 1)
    ranges = []
    for ind in range(split_indexes.size - 1):
        ranges.append((split_indexes[ind], split_indexes[ind + 1]))
    return ranges


def terasort(num_fragments, num_entries, num_buckets, seed):
    """
    ----------------------
    Terasort main program
    ----------------------
    This application generates a set of fragments that contain randomly
    generated key, value tuples and sorts them all considering the key of
    each tuple.

    :param num_fragments: Number of fragments to generate
    :param num_entries: Number of entries (k,v tuples) within each fragment
    :param num_buckets: Number of buckets to consider.
    :param seed: Initial seed for the random number generator.
    """
    from pycompss.api.api import compss_wait_on

    dataset = [gen_fragment(num_entries, seed + i) for i in range(num_fragments)]

    # Init buckets dictionary
    buckets = {}
    for i in range(num_buckets):
        buckets[i] = []

    # Init ranges
    ranges = generate_ranges(num_buckets)
    assert(len(ranges) == num_buckets)

    for d in dataset:
        fragment_buckets = filter_fragment(d, ranges)  # en fragment
        for i in range(num_buckets):
            buckets[i].append(fragment_buckets[i])

    # Verify that the buckets contain elements from their corresponding range
    # This verification can also be done when using PyCOMPSs but will require
    # a synchronization.
    # for i in range(num_buckets):
    #     bucket = buckets[i]
    #     for elem in bucket:
    #         for kv in elem:
    #             assert(kv[0] >= ranges[i][0] and kv[0] < ranges[i][1])

    result = dict()
    for key, value in buckets.items():
        result[key] = combine_and_sort_bucket_elements(*tuple(value))

    for key, value in result.items():
        result[key] = compss_wait_on(value)

    print("*********** FINAL RESULT ************")
    import pprint
    pprint.pprint(result)
    print("*************************************")


if __name__ == "__main__":
    import sys
    import time

    arg1 = sys.argv[1] if len(sys.argv) > 1 else 16
    arg2 = sys.argv[2] if len(sys.argv) > 2 else 50

    num_of_fragments = int(arg1)  # Default: 16
    num_of_entries = int(arg2)    # Default: 50
    # be very careful with the following argument (since it is in a decorator)
    num_of_buckets = 10           # int(sys.argv[3])
    seed = 5

    startTime = time.time()
    terasort(num_of_fragments, num_of_entries, num_of_buckets, seed)
    print("Elapsed Time {} (s)".format(time.time() - startTime))
