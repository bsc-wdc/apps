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
import tensorflow as tf
import random
import numpy as np


@task(returns=list)
def joinTask(a, b):
    c = []
    for i in range(len(a)):
        coordinates = []
        for j in range(len(a[i][0])):
            coordinates = coordinates + [a[i][0][j] + b[i][0][j]]
        group = (coordinates, a[i][1] + b[i][1])
        c.append(group)
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


def getNewCentroids(ps):
    centroides = []
    for i in range(len(ps)):
        cluster = []
        for j in range(len(ps[i][0])):
            cluster = cluster + [ps[i][0][j] / ps[i][1]]
        centroides.append(cluster)
    return centroides


@task(returns=list)
def kmeans_tf(vectors, centroides, k):
    vecs = tf.constant(vectors)
    centr = tf.constant(centroides)
    expanded_vectors = tf.expand_dims(vectors, 0)
    expanded_centroides = tf.expand_dims(centroides, 1)
    assignments = tf.argmin(tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroides)), 2), 0)
    with tf.Session() as sess:
        grouping = []
        for c in range(k):
            nelem = tf.reduce_sum(tf.to_int32(tf.equal(assignments, c)))
            sum_points = tf.reduce_sum(tf.gather(vecs, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])), reduction_indices=[1])
            sum_points = sum_points.eval().tolist()
            component = (sum_points[0], nelem.eval().tolist())
            grouping.append(component)
        return grouping


@task(returns=list)
def genFragments(size, dim):
    return genPoints(size, dim)


def genPoints(size, dim):
    vectors_set = []
    for i in range(size):
        point = []
        if np.random.random() > 0.5:
            for j in range(dim):
                point = point + [np.random.normal(0.0, 0.9)]
        else:
            for j in range(dim):
                point = point + [np.random.normal(1.5, 0.5)]
        vectors_set.append(point)
    return vectors_set


def compute_distance(centroides, means):
    distance = 0
    for i in range(len(centroides)):
        for j in range(len(centroides[i])):
            elem = (centroides[i][j] - means[i][j])
            distance = distance + elem*elem
    return distance


def kmeans_frag(numP, k, maxIterations, numFrag, dim, convergenceFactor):
    from pycompss.api.api import compss_wait_on

    size = int(numP/numFrag)

    X = [genFragments(size, dim) for _ in range(numFrag)]
    centroides = genPoints(k, dim)
    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)
        for j in range(maxIterations):
            print(j)
            groups = [kmeans_tf(X[i], centroides, k) for i in range(numFrag)]
            partialSum = mergeReduce(joinTask, groups)
            partialSum = compss_wait_on(partialSum)
            means = getNewCentroids(partialSum)
            distance = [np.linalg.norm(np.asarray(centroides[i]) - np.asarray(means[i])) for i in range(len(means))]
            distance = sum(distance)
            if distance < convergenceFactor:
                break
            centroides = means
        return means


if __name__ == "__main__":
    import sys
    import time
    import numpy as np

    numP = int(sys.argv[1])
    dim = int(sys.argv[2])
    k = int(sys.argv[3])
    numFrag = int(sys.argv[4])

    startTime = time.time()
    result = kmeans_frag(numP, k, 5, numFrag, dim, 1e-4)
    print(numP)
    print(result)
    print("Elapsed Time {} (s)".format(time.time() - startTime))
