#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
import sys
import numpy as np
import pickle

def init_board_gauss(N, k):
    n = float(N)/k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    return X

N = int(sys.argv[1])
K = int(sys.argv[2])
#N = 100
#K = 7
X = init_board_gauss(N,K)

ff = open('n.txt','w')
pickle.dump(X,ff)
ff.close()

'''
f = open('n.txt','r')
aux = pickle.load(f)
print(aux)
'''