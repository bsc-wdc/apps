#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import *
import pickle


def main():
    numbers = 102400
    maxN = 200000
    dataset_file="dataset.txt"

    nums = random.random_integers(maxN, size=(numbers,))
    # Plain numbers output
    with open(dataset_file, 'w') as dataset:
        for n in nums:
            dataset.write(str(n) + ' ')
    '''
    # Pickled output
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
