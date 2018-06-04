#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
from pycompss.api.task import task
import numpy as np
from functools import reduce


def show(data, transform_data, mean, eig, classes):
    from matplotlib import pyplot as plt
    from .plotaux import Arrow3D
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(111, projection='3d')

    num_points = len(data[0]) / classes
    obj = ['o', 'x', '^']
    for c in range(classes):
        s = c * num_points
        e = s + num_points
        ax.plot(data[0][s:e], data[1][s:e], data[2][s:e], obj[c])

    ax.plot([mean[0]], [mean[1]], [mean[2]], 'o', color='red')
    for n, w in eig:
        v = w.T
        a = Arrow3D([mean[0], v[0] + mean[0]],
                    [mean[1], v[1] + mean[1]],
                    [mean[2], v[2] + mean[2]],
                    mutation_scale=20,
                    lw=3,
                    arrowstyle="-|>",
                    color="r")
        ax.add_artist(a)
    plt.savefig('PCA3dim.png')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    for c in range(classes):
        s = c * num_points
        e = s + num_points
        ax.plot(transform_data[0][s:e], transform_data[1][s:e], obj[c])

    plt.savefig('PCA2dim.png')


def generate_data(num_v, dim, k):
    n = int(float(num_v) / k)
    data = []
    np.random.seed(8)
    cov = np.eye(dim)
    for k in range(k):
        mu = [k] * dim
        data.append(np.random.multivariate_normal(mu, cov, n).T)
    return np.concatenate(([data[i] for i in range(k)]), axis=1)


@task(returns='numpy.float64')
def _mean_vector(sample):
    return np.mean(sample)


def mean_vector(samples):
    m = list(map(_mean_vector, samples))
    return m


def scatter_matrix(samples, mean_vector, dim):
    data = [samples[:, i] for i in range(len(samples[0]))]
    sm = np.zeros((dim, dim))
    for p in data:
        pt = p.reshape(dim, 1)
        sm += (pt - mean_vector).dot((pt - mean_vector).T)
    return sm


@task(returns=list)
def normalize(data, mean):
    return [x - mean for x in data]


@task(returns='numpy.float64')
def dot_product(P, Q):
    val = [p_q[0].dot(p_q[1].T) for p_q in zip(P, Q)]
    sm = reduce(lambda x, y: x + y, val, 0)
    return sm


def scatter_matrix_d(data, mean, dim):
    sm = [[0 for _ in range(dim)] for _ in range(dim)]
    points = []
    for i in range(dim):
        # points.append(map(lambda x: x-mean, data[i]))
        points.append(normalize(data[i], mean))
    for i in range(dim):
        for j in range(dim):
            sm[i][j] = dot_product(points[i], points[j])
            # val = map(lambda (p, q): p.dot(q.T), zip(points[i], points[j]))
            # sm[i][j] += reduce(lambda x, y: x+y, val)
    return sm


# @task(reuturns=list)
def eigen_values(scatter_matrix):
    print(scatter_matrix)
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(len(eig_val))]
    return eig


# @task(returns='numpy.ndarray')
def transform(data, eig, dim):
    eig_sorted = sorted(eig, key=lambda x: x[0], reverse=True)
    w = np.hstack([eig_sorted[i][1].reshape(dim, 1) for i in range(dim - 1)])
    transform_dim = w.T.dot(data)
    return transform_dim


def PCA():
    from pycompss.api.api import compss_wait_on
    num_points = int(sys.argv[1])  # Example: 1000
    dim = int(sys.argv[2])         # Example: 8
    classes = int(sys.argv[3])     # Example: 10

    data = generate_data(num_points, dim, classes)
    print(len(data[0]))
    m = mean_vector(data)

    scatter_matrix = scatter_matrix_d(data, m, dim)

    scatter_matrix = list(map(compss_wait_on, scatter_matrix))

    eig = eigen_values(scatter_matrix)

    transform_dim = transform(data, eig, dim)

    # show(data, transform_dim, m, eig, classes)

if __name__ == "__main__":
    PCA()
