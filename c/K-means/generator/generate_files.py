#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
import sys
import numpy as np
import pickle


# USAGE EXAMPLE:
# python generate_files.py 2000 5 10 4


def generate_data(num_v, dim, k):
    n = int(float(num_v) / k)
    data = []
    random.seed(5)
    for k in range(k):
        c = [random.uniform(-1, 1) for _ in range(dim)]
        s = random.uniform(0.05, 0.5)
        for _ in range(n):
            d = np.array([np.random.normal(c[j], s) for j in range(dim)])
            data.append(d)

    array_data = np.array(data)[:num_v]
    return array_data


def write_to_file(n, k, dim, ind):
    text_file = str("N" + str(n) + "_K" + str(k) + "_d" + str(dim) + "_" + str(ind) + ".txt")
    with open(text_file, 'w') as ff:
        for i in range(len(x)):
            lines = str(i + 1) + ' '
            for j in range(len(x[i])):
                lines += str(x[i][j]) + ' '
            lines += '\n'
            ff.write(lines)


if __name__ == "__main__":
    N = int(sys.argv[1])
    K = int(sys.argv[2])
    dimension = int(sys.argv[3])
    numFrag = int(sys.argv[4])

    nums_per_frag = int(N / numFrag)

    for index in range(numFrag):
        x = generate_data(nums_per_frag, dimension, K)
        print x
        write_to_file(N, K, dimension, index)
