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

import sys
import time
import pickle
from numpy import *
from collections import defaultdict
from collections import OrderedDict
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on


@task(returns=dict)
def calculate_range(block, fragments, num_range, idx):
    step = num_range / fragments
    idx2 = 0
    partial_result = {}
    for num in block:
        initial = 0
        final = step
        while initial < num_range:
            if num >= initial and num < final:
                partial_result[initial, idx, idx2] = num
            initial += step
            final += step
        idx2 += 1
    return partial_result


@task(returns=set)
def reduce(result):
    return set().union(*result)


def sorting(partial_result, result, fragments, num_range):
    new_dict = minimize_dict(partial_result)
    step = num_range / fragments
    sorted_new_dict = OrderedDict(sorted(iter(new_dict.items()),
                                         key=lambda k_v: k_v[0], reverse=False))
    initial = 0
    final = step
    sizes = []
    while initial < num_range:
        start_pos = 0
        for k, v in sorted_new_dict.items():
            sizes.append(start_pos)
            start_pos += len(v)
        position = 0
        for k, v in sorted_new_dict.items():
            if int(k) >= initial and int(k) < final:
                result.append(sort_in_worker(v, sizes[position]))
                initial += step
                final += step
            position += 1
    for i in range(len(result)):
        result[i] = compss_wait_on(result[i])
    sort = reduce(result)
    return sort


@task(returns=dict)
def sort_in_worker(block, position):
    local_block = sorted(block)
    result = {}
    idx = 0
    for num in local_block:
        curr_pos = position + idx
        result[curr_pos] = num
        idx += 1
    return result


def minimize_dict(partial_result):
    new_dict = defaultdict(list)
    for key, value in list(partial_result.items()):
        value = compss_wait_on(value)
        for val in value:
            new_dict[val[0]].append(value[val])
    return new_dict


def sort(nums_file, fragments, num_range):
    from pycompss.api.api import compss_barrier

    # Read nums from file
    f = open(nums_file, 'r')

    dataset = []
    for line in f:
        dataset.append(asarray([int(x) for x in line.strip().split(" ")]))

    # Flat dataset
    nums = [item for sublist in dataset for item in sublist]

    nums = len(nums)

    if nums / fragments < num_range:
        print('ERROR: num_range should be greater than nums/fragments')
        num_range = int(nums / fragments)
        print("Using num_range: %s" % num_range)

    nums_per_node = nums / fragments
    partial_result = {}
    result = []

    start = time.time()

    keys = [nums[i * nums_per_node:i * nums_per_node + nums_per_node]
            for i in range(fragments)]

    idx = 0
    for k in keys:
        partial_result[idx] = calculate_range(k, fragments, num_range, idx)
        idx += 1

    sorted_nums = sorting(partial_result, result, fragments, num_range)

    compss_barrier()

    print("Elapsed time(s)")
    print(time.time() - start)

    print(sorted_nums)

    '''
    # Save result file
    sorted_nums = compss_wait_on(sorted_nums)
    aux = list(sorted_nums.items())
    result_file = open('./result.txt','w')
    pickle.dump(aux, result_file)
    result_file.close()
    '''


if __name__ == "__main__":
    nums_file = sys.argv[1]
    fragments = int(sys.argv[2])
    num_range = int(sys.argv[3])

    sort(nums_file, fragments, num_range)
