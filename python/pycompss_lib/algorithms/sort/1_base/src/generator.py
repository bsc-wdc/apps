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

from numpy import *


def main():
    nums = 102400
    max_n = 200000
    dataset_file = "dataset.txt"

    nums = random.random_integers(max_n, size=(nums,))
    # Plain nums output
    with open(dataset_file, 'w') as dataset:
        for n in nums:
            dataset.write(str(n) + ' ')
    '''
    # Pickled output
    import pickle

    print("Nums: %s" % str(nums))
    ff = open(dataset_file, 'w')
    pickle.dump(nums, ff)
    ff.close()

    f = open(dataset_file, 'r')
    aux = pickle.load(f)
    print(aux)
    '''


if __name__ == '__main__':
    main()
