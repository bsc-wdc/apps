#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from pycompss.api.task import task
from pycompss.api.parameter import *

@task(returns = dict)
def cluster_points_partial(XP,mu,ind):
	dic = {}
	for x in enumerate(XP):
		bestmukey = min([(i[0], np.linalg.norm(x[1]-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
		if bestmukey not in dic:
			dic[bestmukey] = [x[0]+ind]
		else:
			dic[bestmukey].append(x[0]+ind)
	return dic

@task(returns = dict)
def partial_sum(XP, clusters,ind):
	print clusters
	p = [(i,[(XP[j-ind]) for j in clusters[i]]) for i in clusters]
	print p
	dic = {}
	for i,l in p:
		dic[i] = (len(l),np.sum(l,axis = 0))
	return dic

@task(returns = list)
def reduceCenters(partialResult):
	centers = {}
	for l in partialResult:
		for key in l:
			if key not in centers:
				centers[key] = l[key]
			else:
				centers[key] = (centers[key][0]+l[key][0],centers[key][1]+l[key][1])
	
	return [centers[c][1]/centers[c][0] for c in centers]

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
    if oldmu != []:
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

def kmeans_frag(X, k, epsilon, maxIterations,numFrag):
	import random
	import numpy as np
	from pycompss.api.api import compss_wait_on

	mu = random.sample(X,k)
	oldmu = []
	n=0
	size = int(len(X)/numFrag)
	while not has_converged(mu, oldmu,epsilon,n,maxIterations):
		oldmu = mu
		clusters = [cluster_points_partial(X[f*size:f*size+size],mu,f*size) for f in range(numFrag)]
		partialResult = [partial_sum(X[f*size:f*size+size],clusters[f],f*size) for f in range(numFrag)]
		
		#partialResult = compss_wait_on(partialResult)
		
		mu = reduceCenters(partialResult)
		mu = compss_wait_on(mu)
		print mu
		n += 1
	return n

if __name__ == "__main__":
	import sys
	import random

	numV = int(sys.argv[1])
	dim = int(sys.argv[2])
	k = int(sys.argv[3])
	numFrag = int(sys.argv[4])

	X = generate_data(numV,dim,k)
	result = kmeans_frag(X,k,1e-8,10,numFrag)
	print result


