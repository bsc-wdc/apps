#!/usr/bin/python
import numpy as np
import random
import pickle
import sys
import time

from pycompss.api.task import task
from pycompss.api.parameter import *


@task(clusters=INOUT)
def assign(clusters,passign):
    for p in passign:
        x, b = p
        try:
            clusters[b].append(x)
        except KeyError:
            clusters[b] = [x]

@task(clusters = INOUT)
def assign_sorted(clusters,passign,i):
    for p in passign[i]:
        try:
            clusters.append(p)
        except KeyError:
            clusters = [p]

@task(returns = list)
def cluster_points(XP, mu):
    pb  = []
    for x in XP:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]

        pb.append((x,bestmukey))
    return pb


#@task(returns = list, priority=True)
@task(returns = list)
def cluster_points_sorted(XP, mu,K):
    pb  = [[]for k in range(K)]
    
    for x in XP:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]

        pb[bestmukey].append(x)
    return pb

@task(returns = list) 
def reevaluate_centers(clusters):
    newmu = []
    #keys = sorted(clusters.keys())
    for k in range(len(clusters)):
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu

@task(clusters = INOUT, returns = list)
def reevaluate_centers_list(lista,clusters):
    #clusters = {}
    for passign in lista:
        for p in passign:
            x, b = p
            try:
                clusters[b].append(x)
            except KeyError:
                clusters[b] = [x]
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu

def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

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



if __name__ == "__main__":
    import sys
    from pycompss.api.api import compss_wait_on
    #from kmeans.experiment_params_800 import nList,kList,n,k

    #print nList
    
    rfile = sys.argv[1]
    
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    numFrag = int(sys.argv[4])
    
    ''' Read points from file'''
    #pfile = sys.argv[5]
    #f = open(pfile,'r')
    #X = pickle.load(f)

    '''Generate Points'''
    X = init_board_gauss(N,K)
    print ("X= ",X)

    '''hiBench'''
    #from kmeans.experiment_params_800 import nList,kList,n,k
    #print type(X)
    #X = np.asarray(nList)
    #K = k
    #N = n
    #mu = np.asarray(kList)
    #print ("N: ",N)
    #print X
    size = len(X)/numFrag
    rs = len(X)%numFrag
    
    start = time.time()
    
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    print ("oldmu= ",oldmu)
    print ("mu= ",mu)
    n=0
    while not has_converged(mu, oldmu):
        n = n+1
        oldmu = mu
        passign = [[]for i in range(numFrag)]
        endidx = 0
        clusters = [[]for k in range(K)]
        for block_i in range(numFrag):
            startidx = endidx
            endidx = startidx+size
            print (startidx, endidx)
            if rs > 0: 
                endidx = endidx + 1
                rs = rs - 1
            # Assign all points in X to clusters
            passign[block_i] = cluster_points_sorted(X[startidx:endidx+1], mu,K)
            #for s in range(len(clusters)):
            for s in range(K):
                assign_sorted(clusters[s],passign[block_i],s)
                clusters[s] = compss_wait_on(clusters[s])
        # Reevaluate centers
        #clusters = compss_wait_on(clusters)
        mu = reevaluate_centers(clusters)
        mu = compss_wait_on(mu)
        print n

    #clusters = compss_wait_on(clusters)

    end = time.time()
    print "Ellapsed time(s)" 
    print end - start
    print "Iterations"
    print n
    

    print "FINAL"
    
    ff = open(rfile,'w')
    pickle.dump(mu,ff)
    ff.close()
    
    '''
    import matplotlib.pyplot as plt
    x,y = zip(*mu)

    points = [zip(*clusters[i]) for i in range(len(clusters))]
    plt.plot(x,y,'bx')
    for i in range(len(points)):
        plt.plot(points[i][0],points[i][1],'o')
    plt.autoscale(enable=True, axis='both', tight=False)
    plt.savefig(rfile)
    '''
