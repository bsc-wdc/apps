#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
import sys
import numpy as np
import pickle


# USAGE EXAMPLE:
# python generate_files.py 2000 5 10 4


def generate_data(numV, dim, K):
    n = int(float(numV) / K)
    data = []
    random.seed(5)
    for k in range(K):
        c = [random.uniform(-1, 1) for i in range(dim)]
        s = random.uniform(0.05, 0.5)
        for i in range(n):
            d = np.array([np.random.normal(c[j], s) for j in range(dim)])
            data.append(d)

    Data = np.array(data)[:numV]
    return Data

if __name__ == "__main__":
    N = int(sys.argv[1])
    K = int(sys.argv[2])
    dim = int(sys.argv[3])
    numFrag = int(sys.argv[4])

    nums_per_frag = int(N/numFrag)

    for ind in range(numFrag):
        x = generate_data(nums_per_frag, dim, K)
        print x
        textFile = str("N"+str(N)+"_K"+str(K)+"_d"+str(dim)+"_"+str(ind)+".txt")
        ff = open(textFile, 'w')
        for i in range(len(x)):
            lines = ""
	    lines += str(i+1)+' '
            for j in range(len(x[i])):
                lines += str(x[i][j])+' '
            lines += '\n'
            ff.write(lines)
        ff.close()


