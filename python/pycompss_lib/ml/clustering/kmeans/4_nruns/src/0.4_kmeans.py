#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from pycompss.api.task import task
from pycompss.api.parameter import *

def cluster_points(XP,mu):
	dic = {}
	for x in enumerate(XP):
		bestmukey = min([(i[0], np.linalg.norm(x[1]-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
		if bestmukey not in dic:
			dic[bestmukey] = [x[0]]
		else:
			dic[bestmukey].append(x[0])
	return dic


def reevaluate_centers(XP,clusters):
	p = [[XP[j] for j in clusters[i]] for i in clusters]
	return [np.mean(l,axis = 0) for l in p]

def generate_data(numV, dim, K):
    n = int(float(numV)/K)
    data = []

    for k in range(K):
        c = [random.uniform(-1, 1) for i in range(dim)]
        s = random.uniform(0.05,0.5)
        for i in range(n):
            d = np.array([np.random.normal(c[j],s) for j in range(dim)])
            data.append(d)

    Data = np.array(data)[:numV]
    return Data

def has_converged(mu, oldmu, epsilon,iter,maxIterations):
    print "iter: "+str(iter)
    print "maxIterations: "+str(maxIterations)
    if iter < maxIterations:
        aux = [np.linalg.norm(oldmu[i]-mu[i]) for i in range(len(mu))]
        distancia = sum(aux)
        if distancia < epsilon*epsilon:
            print "Distancia_T: "+str(distancia)
            return True
        else:
            print "Distancia_F: "+str(distancia)
            return False
    else:
        # detencion pq se ha alcanzado el maximo de iteraciones
        return True

@task(returns = int)
def kmeans(X, k, epsilon, maxIterations):
	import random
	import numpy as np
	mu = random.sample(X,k)
	n=0
	first = True
	while first or not has_converged(mu, oldmu,epsilon,n,maxIterations):
		if first:
			first = False
		oldmu = mu
		clusters = cluster_points(X,mu)
		mu = reevaluate_centers(X, clusters)
		n += 1
	return n

if __name__ == "__main__":
	import sys
	import random
	from pycompss.api.api import compss_wait_on
	numV = int(sys.argv[1])
	dim = int(sys.argv[2])
	k = int(sys.argv[3])
	r = int(sys.argv[4])
	
	data = [generate_data(numV,dim,k) for i in range(r)]
	result = [kmeans(data[i],k,1e-8,10) for i in range(r)]
	#X = generate_data(numV,dim,k)
	#result = kmeans(X,k,1e-8,10)
	
	result = compss_wait_on(result)
	print result


