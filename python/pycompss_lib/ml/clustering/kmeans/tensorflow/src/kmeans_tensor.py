#from pycompss.api.task import task
#from pycompss.api.parameter import * 
import tensorflow as tf 
import random 
import numpy as np 
#import pandas as pd 
#import seaborn as sns   
#import matplotlib.pyplot as plt  



@task(returns=object)
def joinTask(a,b):
	c = tf.add(a,b)
	return c


def mergeReduce(function, data):
    from collections import deque
    q = deque(range(len(data)))
    while len(q):
        x = q.popleft()
        if len(q):
            y = q.popleft()
            data[x] = function(data[x], data[y])
            q.append(x)
        else:
            return data[x]


def getNewCentroids(ps,dim):
	points = tf.slice(ps,[0,0],[-1,dim])
	nelems = tf.slice(ps,[0,dim],[-1,dim])
	return tf.div(points,nelems)


def combine(vectors,assignments,c,dim):
	nelem = tf.reduce_sum(tf.to_int64(tf.equal(assignments,c)))
	nelem = tf.to_float(tf.tile(tf.reshape(nelem,[-1]),[dim]))
	sum_points = tf.reduce_sum(tf.gather(vectors,tf.reshape(tf.where(
				tf.equal(assignments,c)),[1,-1])),reduction_indices=[1])
	sum_points = tf.squeeze(sum_points)
	component = tf.concat(0,[sum_points,nelem])
	return component


@task(returns=object)
def kmeans_tf(vectors,centroides,k,dim):
	expanded_vectors = tf.expand_dims(vectors,0) 									#SHAPE: [1,size,dim]
	expanded_centroides = tf.expand_dims(centroides,1) 								#SHAPE: [k,1,dim]
	assignments = tf.argmin(tf.reduce_sum(tf.square(tf.sub(expanded_vectors,
		expanded_centroides)),2),0)
	pack = tf.pack([combine(vectors,assignments,c,dim) for c in range(k)])
	return pack


@task(returns=object)
def genFragments(size,dim,const):
	vectors_set = []
	for i in range(size):
		point = []
		if np.random.random() > 0.5:
			for j in range(dim):
				point = point + [np.random.normal(0.0,0.9)]
		else:
			for j in range(dim):
				point = point + [np.random.normal(1.5,0.5)]
		vectors_set.append(point)
	if const == True:
		return tf.constant(vectors_set)
	else:
		return tf.Variable(vectors_set)


def kmeans_frag(numP,k,maxIterations,numFrag,convergenceFactor,dim):
	from pycompss.api.api import compss_wait_on
	import time 

	size = int(numP/numFrag)
	X = [genFragments(size,dim,True) for _ in range(numFrag)] 						#SHAPE: array of [size,dim]
	centroides = genFragments(k,dim,False) 												#SHAPE: [k,dim]
	groups = [kmeans_tf(X[i],centroides,k,dim) for i in range(numFrag)] 			#SHAPE: array of [k,2*dim]
	partialSum = mergeReduce(joinTask,groups) 										#SHAPE: [k,2*dim]
	partialSum = compss_wait_on(partialSum)
	means = getNewCentroids(partialSum,dim) 										#SHAPE: [k,dim]
	distance = tf.reduce_sum(tf.square(tf.sub(centroides,means)))
	update_centroides = tf.assign(centroides,means)
	init_op = tf.initialize_all_variables()

	sess = tf.Session() 
	sess.run(init_op)

	for step in range(maxIterations):
		_,centroid_values = sess.run([
		update_centroides,centroides])
		if (distance.eval(session=sess) < convergenceFactor):
			break

	return centroides.eval(session=sess)


if __name__ == "__main__":
	import sys
	import time 
	import numpy as np 

	numP = int(sys.argv[1])
	dim = int(sys.argv[2])
	k = int(sys.argv[3])
	numFrag = int(sys.argv[4])

	startTime = time.time()
	result = kmeans_frag(numP,k,10,numFrag,1e-4,dim)
	print(result)
	print("Ellapsed Time {} (s)".format(time.time() - startTime))

