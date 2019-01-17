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

from pycompss.api.task import task
import sys

range_min = 0
range_max = sys.maxsize


@task(returns=list)
def gen_fragment(num_entries, _seed):
    """
    Generate a fragment with random numbers.
    A fragment is a list of tuples, where the first element of each tuple
    is the key, and the second the value.
    Each key is generated randomly between min and max global values.
    Each value is generated randomly between -1 and 1

    fragment structure = [(k1, v1), (k2, v2), ..., (kn, vn)]

    :param num_entries: Number of k,v pairs within a fragment
    :param _seed: The seed for the random generator
    :return: Fragment
    """
    import random
    random.seed(_seed)
    fragment = []
    for n in range(num_entries):
        fragment.append((random.randrange(range_min, range_max), random.random()))
    return fragment


@task(returns=tuple([tuple() for i in range(10)]))  # Multireturn
def filter_fragment(fragment, ranges):
    """
    Task that filters a fragment entries for the given ranges.
        * Ranges is a list of tuples where each tuple corresponds to
          a range.
        * Each tuple (range) is composed by two elements, the minimum
          and maximum of each range.
        * The filtering is performed by checking which fragment entries'
          keys belong to each range.
    The entries that belong to each range are considered a bucket.
        * The variable buckets is a list of lists, where the inner lists
          correspond to the bucket of each range.

    :param fragment: The fragment to be sorted and filtered.
    :param ranges: The ranges to apply when filtering.
    :return: Multireturn of the buckets.
    """
    buckets = []
    for _range in ranges:
        buckets.append([k_v for k_v in fragment if
                        _range[0] <= k_v[0] < _range[1]])
    return tuple(buckets)


@task(returns=dict)
def combine_and_sort_bucket_elements(*args):
    """
    Task that combines the buckets received as *args parameter and final
    sorting.

    *args structure = ([],[], ..., [])

    :param args: *args that contains the buckets of a single range
    :return: A list of tuples with the same format as provided initially
             sorted by key.
    """
    combined = []
    for e in args:
        for kv in e:
            combined.append(kv)
    sorted_by_key = sorted(combined, key=lambda key: key[0])
    return sorted_by_key
