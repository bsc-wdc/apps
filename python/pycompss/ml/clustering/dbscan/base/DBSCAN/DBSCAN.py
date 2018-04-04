#
#  Copyright 2.02-2017 Barcelona Supercomputing Center (www.bsc.es)
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
"""
@author: csegarra

PyCOMPSs Mathematical Library: Clustering: DBSCAN
=================================================
    This file contains the DBSCAN algorithm.
"""

import itertools
import time
import numpy as np
import sys
import math
from classes.cluster import Cluster
from classes.DS import DisjointSet
from collections import defaultdict

from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.api.api import compss_wait_on
 
def normalizeData(dataFile):
    """
    Given a dataset, divide each dimension by its maximum
    :param dataFile: path to the original .txt
    :return newName: new name of the data file containing the normalized data.
    """
    dataset = np.loadtxt(dataFile)
    normData = np.where(np.max(dataset, axis=0)==0, dataset, dataset*1./np.max(dataset, axis=0))
    newName = dataFile[dataFile.find('.')+1:dataFile.rfind('.')]
    newName = '.'+newName+'n.txt'
    np.savetxt(newName,normData)
    return newName

def partitionSpace(dataset, fragSize, epsilon):
    """
    Gives a space partition basing on point density
    :returns fragData:      dict with (key,value)=(spatial id, points in the region)
    :returns rangeToEps:    list with the #squares until epsilon distance 
    """
    fragData = defaultdict(list)
    rangeToEps = defaultdict(list)
    dim = len(dataset[0])
    fragVec = [[np.max(np.min(dataset, axis=0)[i]-epsilon,0), np.mean(dataset,axis=0)[i],                   np.max(dataset, axis=0)[i]] for i in range(dim)]
    size = pow(10,len(str(fragSize+1)))
    for i in range(dim):
        for j in range(fragSize):
            tmpPoint = defaultdict(int)
            for point in dataset:
                k = 0
                while point[i] > fragVec[i][k]: k += 1
                tmpPoint[k] += 1        
            ind = max(tmpPoint.iterkeys(), key = (lambda key: tmpPoint[key]))
            val = float((fragVec[i][ind-1] + fragVec[i][ind])/2)
            fragVec[i].insert(ind, val)
    for point in dataset:
        key = 0
        for i in range(dim):
            k = 0
            while point[i] > fragVec[i][k]: k += 1
            key += (k-1)*pow(size,i)        
        fragData[key].append(point)
    for square in fragData:
        tmp = []
        for i in range(dim):
            pos=square%size
            a = [[j,x-fragVec[i][pos]] for j,x in enumerate(fragVec[i]) if abs(x-fragVec[i][pos])                   < epsilon]
            b = [[j,x-fragVec[i][pos+1]] for j,x in enumerate(fragVec[i]) if abs(x-fragVec[i][pos+                  1]) < epsilon]
            maxa = abs(max(a, key = lambda x: x[1])[0] - pos)
            maxb = abs(max(b, key = lambda x: x[1])[0] - pos)
            tmp.append(max(maxa, maxb, 1)) 
            pos = pos/size
        rangeToEps[square] = tmp
    return (fragData, fragVec, rangeToEps)

def partialScan(corePoints, square, epsilon, minPoints, fragData, fragSize, numParts, rangeToEps):
    """
    Looks for all the core points (over minPoints neighbours) inside a certain square.
    :inout corePoints:  list were core points found are appended.
    :param square:      space region where core points are looked for.
    :param fragData:    dict containing space partition and its points.
    :param rangeToEps:  for each square, number of neighbors until epsilon distance. 
    """
    dim = len(fragData[square][0])
    pointSet = fragData[square]
    pointSetReal = pointSet[:]
    k=rangeToEps[square]
    size = pow(10,len(str(fragSize + 1)))
    perm = []
    for i in range(dim):
        perm.append(range(-k[i],k[i] + 1))
    for comb in itertools.product(*perm):
        current = square
        for i in range(dim):
            current = current+comb[i]*math.pow(size,i)
        if current in fragData and current != square:
            pointSet = pointSet+fragData[current]
    for i in range(numParts):
        tmpPS = [p for j,p in enumerate(pointSetReal) if j % numParts == i]
        scanTask(corePoints[i], pointSet, tmpPS, epsilon, minPoints)

@task(clusters = INOUT) 
def mergeCluster(clusters, corePoints, square, epsilon):
    """
    Append a list of core points to the current clusters list.
    :inout clusters:    curent cluster list.
    :param corePoints:  list of all the core points found in square.
    :param square:      working square.
    """
    for point in corePoints:
        possibleClusters = []
        for clust in clusters:
            for clustP in clust.points:
                if np.linalg.norm(point - clustP) < epsilon:
                    possibleClusters.append(clust)
                    if len(possibleClusters) > 1:
                        clusters.remove(clust)
                    break
        if len(possibleClusters) > 0:
            master = possibleClusters.pop(0)
            master.addPoint(point)
            for slave in possibleClusters:
                master.merge(slave)
        else:
            tmp = ([point],[square])
            tmpc = Cluster()
            tmpc.add(*tmp)
            clusters.append(tmpc)

def syncClusters(clusters, epsilon, numParts):
    """
    Returns a matrix of booleans. Pos [i,j]=1 <=> clusters -i and -j should be merged.
    :inout clusters:            list of all clusters and their points.
    :param numComp:             number of comparisons per worker (i.e tasks).
    :return possibleClusters:   adjacency matrix.
    """
    possibleClusters = defaultdict(list)
    numComp = 10
    tmpDel = (numComp-len(clusters)%numComp)%numComp
    if not len(clusters)%numComp == 0:
        for i in range(numComp - len(clusters)%numComp): clusters.append(Cluster())
    enable = [[[] for z in range(len(clusters)/numComp)] for w in range(len(clusters)/numComp)]
    for i in range(len(clusters)/numComp):
        tmp1 = [clusters[numComp*i + k].points for k in range(numComp)]
        for j in range(len(clusters)/numComp):
            tmp2 = [clusters[numComp*j + k].points for k in range(numComp)]
            syncTask(enable[i][j], tmp1, tmp2, epsilon)
    enable = compss_wait_on(enable)
    for i in range(len(clusters)/numComp): 
        for j in range(len(clusters)/numComp):
            for k in range(numComp):
                for l in range(numComp):
                    if enable[i][j][numComp*k + l]: 
                        possibleClusters[i*numComp + k].append(j*numComp + l)
    l = len(clusters)
    for i in range(tmpDel):
        possibleClusters.pop(l - 1 - i, None)
        clusters.pop()    
    return possibleClusters

@task(enable = INOUT)
def syncTask(enable, hosts, visits, epsilon):
    """
    Given two lists, checks wether the distance between the two sets is less than epsilon.
    """
    for host in hosts:
        for visit in visits:
            for p in host:
                for point in visit:
                    if np.linalg.norm(point - p) < epsilon:
                        enable.append(1)
                        break
                else: continue
                break   
            else: enable.append(0)

@task(corePoints = INOUT) 
def scanTask(corePoints, pointSet, pointSetReal, epsilon, minPoints):
    """"
    Given a list of core points and a point set, finds all the core points.
    :param pointSetReal:    set of points were core points are looked for.
    :param pointSet:        set of points were all the neighbors might be.
    """
    for point in pointSetReal:
        neighbourPts = 0
        for p in pointSet:
            if np.linalg.norm(point-p) < epsilon:
                neighbourPts = neighbourPts+1
                if neighbourPts >= minPoints:
                    corePoints.append(point)
                    break  

def expandCluster(clusters, fragData, epsilon, minPoints, fragSize, numParts, rangeToEps):
    """
    Expands all clusters contained in a list of clusters. 
    already established clusters.
    """
    addToClust = [[[] for _ in range(numParts)] for x in range(len(clusters))]
    for numClust,clust in enumerate(clusters):
        for k in range(numParts):
            neighExpansion(addToClust[numClust][k], clust, fragData, fragSize, rangeToEps, epsilon)
    addToClust = compss_wait_on(addToClust)
    for i,m in enumerate(addToClust):
        addToClust[i] = [j for k in m for j in k]
    pointsToClust = defaultdict(list)
    links = defaultdict(list)
    for i,clust in enumerate(addToClust):
        for p in addToClust[i]:
            if str(p) in pointsToClust:
                for c in pointsToClust[str(p)]:
                    if not i in links[c]: links[c].append(i)
                    if not c in links[i]: links[i].append(c)
                pointsToClust[str(p)].append(i)
            else:
                pointsToClust[str(p)].append(i)
                clusters[i].addPoint(p) 
    return update(clusters, links, False)
            
@task(clustPoints = INOUT)
def neighExpansion(clustPoints, clust, fragData, fragSize, rangeToEps, epsilon):
    """
    Given a cluster of core points, returns all the points lying in the squares reachable.
    :inout clustPoints: list where possible neighbor points will be added.
    :param clust: cluster whose neighbour points will be added.
    """
    squaresNot = clust.square[:]
    pointSet = []
    for sq in clust.square:
        #Segur que no ho sumem dos cops aleshores?
        pointSet += fragData[sq][:]
        dim = len(fragData[sq][0])    
        k = rangeToEps[sq]
        size = pow(10,len(str(fragSize + 1)))
        perm = []
        for i in range(dim):
            perm.append(range(-k[i],k[i] + 1))
        for comb in itertools.product(*perm):
            current = sq
            for i in range(dim):
                current = current + comb[i]*math.pow(size,i)
            if current in fragData and not (current in squaresNot):
                pointSet = pointSet + fragData[current][:]
                squaresNot.append(current)
    tmp = [b for b in pointSet if not (b in clust.points)]
    for i,point in enumerate(tmp):
        for p in clust.points:
            if np.linalg.norm(point - p) < epsilon:
                clustPoints.append(point)

def update(clusters, possibleClusters, returnCluster):
    """
    Updates the clusters given an adjacency matrix.
    :param clusters:            provisional cluster list.
    :param possibleClusters:    adjacency matrix.
    :param returnCluster:       indicates wether the result must mantain the cluster struc.
    :return defClusters:        updated cluster list.
    """
    MF_set = DisjointSet(range(len(clusters)))
    for i in possibleClusters:
        for j in range(len(possibleClusters[i]) - 1):
            MF_set.union(possibleClusters[i][j], possibleClusters[i][j + 1])
    a = MF_set.get()
    if returnCluster:
        defCluster = [Cluster() for _ in range(len(a))]
        for i,lst in enumerate(a):
            for elem in lst:
                defCluster[i].merge(clusters[elem])
        return defCluster
    defCluster = [[] for _ in range(len(a))]
    for i,lst in enumerate(a):
        for elem in lst:
            for point in clusters[elem].points:
                defCluster[i].append(point)
    return defCluster

def DBScan(dataFile, fragSize, epsilon, minPoints, numParts):
    """
    Main DBSCAN algorithm.
    :param dataFile:    path to the dataset.    
    :param epsilon:     maximum distance for two points to be considered neighbours.
    :param minPoints:   minimum number of neighbours for a point to be considered core point.
    :param fragData:    dict with the partitioned space and the points classified.
    :param fragSize:    size used for space partition.
    :param numParts:    number of parts in which fragData is divided for processing.
    :return defClusters:list of the final clusters.
    :return fragVec:    object used for the plotting.
    """
    print "Density Based Scan started."
    start = time.time()
    normData = normalizeData(dataFile) 
    dataset = np.loadtxt(normData)
    [fragData, fragVec, rangeToEps] = partitionSpace(dataset, fragSize, epsilon) 
    print "Starting partial scan..."
    clusters = [[[] for _ in range(len(fragData))] for __ in range(numParts)]
    corePMatrix = [[[] for _ in range(numParts)] for __ in range(len(fragData))]
    for i, (square,value) in enumerate(fragData.viewitems()):
        partialScan(corePMatrix[i], square, epsilon, minPoints, fragData, fragSize, numParts,                   rangeToEps)
        for j in range(numParts):
            mergeCluster(clusters[j][i], corePMatrix[i][j], square, epsilon)
    clusters = compss_wait_on(clusters)
    print "Initial Proposal Finished"
    iniP = time.time()
    clusters = [clust for rows in clusters for squares in rows for clust in squares] 
    print "Length of clusters found: "+str(len(clusters))
    possibleClusters = syncClusters(clusters, epsilon, numParts)
    syncT = time.time()
    print "Syncing Finished"
    halfDefClusters = update(clusters, possibleClusters, True)
    updateTime = time.time()
    defClusters = expandCluster(halfDefClusters, fragData, epsilon, minPoints, fragSize, numParts,          rangeToEps)
    end = time.time()
    print "Expansion finished"
    print "DBSCAN Algorithm finished succesfully."
    print "Exec Times:"
    print "----------------------------------"
    print "Initial Proposal Time: \t %.6f" % float(iniP-start)
    print "Syncing Time: \t \t  %.6f" % float(syncT-iniP)
    print "Update Time: \t \t  %.6f" % float(updateTime-syncT)
    print "Expand: \t \t  %.6f" % float(end-updateTime)
    print "----------------------------------"
    print "Time elapsed: \t \t  %.6f" % float(end-start)
    print "Number of clusters found: "+str(len(defClusters))
    sys.stdout.flush()
    sys.stderr.flush()
    return [defClusters, fragVec]
