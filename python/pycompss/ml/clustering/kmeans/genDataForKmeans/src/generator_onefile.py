#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
import sys
import numpy as np
import pickle

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
    dim =int(sys.argv[3])
    
    X = generate_data(N,dim,K)

    '''Write pickle file'''
    textFile = str("N"+str(N)+"_K"+str(K)+"_d"+str(dim)+".txt")
    ff = open(textFile,'w')
    pickle.dump(X,ff)
    ff.close()

    #'''Read pickle file'''
    #f = open(textFile,'r')
    #aux = pickle.load(f)
    #print(aux)
    
