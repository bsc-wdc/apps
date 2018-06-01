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
from pycompss.api.parameter import *


@task(file=FILE_INOUT)
def read_data(path):
    """
    Read the data from a given file.
    :param path: Input file path.
    :return: Content of the input file.
    """
    f = open(path, 'r')
    import pickle
    data = pickle.load(f)
    f.close()
    return data


@task(file=FILE_IN, returns=int)
def sort_partition(data):
    """ Sorts data, which is assumed to consists of (key, value) tuples list.
    :param data: List of tuples to be sorted.
    :return: sorted list of tuples.
    """
    res = np.sort(data, kind='mergesort')
    return len(res)


def sort_by_key(files_paths, n_times):
    """ Sort by key.
    :param files_paths: List of paths of the input files.
    :param n_times: Number of times to do the sort by key.
    :return: Length of the list of elements sorted.
    """
    from pycompss.api.api import compss_wait_on
    fo_list = []
    dataset = list(map(read_data, files_paths))
    for i in range(n_times):
        fo_list.append(list(map(sort_partition, dataset)))
    result_list = compss_wait_on(fo_list)
    return len(result_list)


def main():
    import sys
    import os
    import time
    path = sys.argv[1]
    n_times = int(sys.argv[2])

    files_paths = []
    for f in os.listdir(path):
        files_paths.append(path + '/' + f)

    start_time = time.time()
    result = sort_by_key(files_paths, n_times)

    print("Elapsed Time(s)")
    print(time.time() - start_time)
    print(result)


if __name__ == "__main__":
    main()
