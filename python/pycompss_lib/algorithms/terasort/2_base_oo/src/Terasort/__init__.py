#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#  Copyright 2002-2015 Barcelona Supercomputing Center (www.bsc.es)
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


class Bucket(object):
    def __init__(self):
        self.unsortedFragments = list()
        self.sortedByKey = []

    @task()
    def addUnsortedFragment(self, fragment):
        """
        Task to append new chunks to a bucket
        :param fragment: unsorted fragment to be included
        """
        self.unsortedFragments.append(fragment)

    @task()
    def combineAndSortBucketElements(self):
        """
        Task that combines the buckets received as *args parameter and final
        sorting.

        *args structure = ([],[], ..., [])

        :param args: *args that contains the buckets of a single range
        :return: A list of tuples with the same format as provided initially
                 sorted by key.
        """
        combined = []
        for entries in self.unsortedFragments:
            for kv in entries:
                combined.append(kv)
        self.sortedByKey = sorted(combined, key=lambda key: key[0])
        #self.sortedByKey = sorted((kv for kv in entries for entries in self.unsortedFragments), key=lambda key: key[0])

    def removeUnsortedFragmentsList(self):
        self.unsortedFragments = list()

    def __repr__(self):
        return self.sortedByKey.__str__()


class Fragment(object):
    def __init__(self, range_min, range_max):
        self.num_entries = 0
        self.entries = list()
        self.range_min = range_min
        self.range_max = range_max

    @task(returns=list)
    def genFragment(self, numEntries, seed):
        """
        Generate a fragment with random numbers.
        A fragment is a list of tuples, where the first element of each tuple
        is the key, and the second the value.
        Each key is generated randomly between min and max global values.
        Each value is generated randomly between -1 and 1

        fragment structure = [(k1, v1), (k2, v2), ..., (kn, vn)]

        :param numEntries: Number of k,v pairs within a fragment
        :param seed: The seed for the random generator
        :return:
        """
        import random
        random.seed(seed)
        fragment = []
        for n in range(numEntries):
            fragment.append((random.randrange(self.range_min, self.range_max),
                             random.random()))
        self.num_entries = numEntries
        self.entries = fragment

    # @task(returns="tuple([tuple() for i in range(len(fragment))])")  # not supported
    @task(returns=tuple([tuple() for i in range(10)]))  # Multireturn
    def filterFragment(self, ranges):
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
        for range in ranges:
            buckets.append(filter(lambda (k, v):
                                  k >= range[0] and k < range[1],
                                  self.entries))
        return tuple(buckets)


class Board(object):
    def __init__(self, range_min, range_max):
        self.num_fragments = 0
        self.fragments = list()
        self.num_entries = 0
        self.range_min = range_min
        self.range_max = range_max

    def init_random(self, num_fragments, num_entries, base_seed):
        self.num_fragments = num_fragments
        self.num_entries = num_entries
        self.base_seed = base_seed

        seed = base_seed
        for i in range(self.num_fragments):
            newFrag = Fragment(self.range_min, self.range_max)
            newFrag.genFragment(num_entries, seed)
            self.fragments.append(newFrag)
            seed += 1
