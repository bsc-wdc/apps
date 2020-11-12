#!/usr/bin/python
#
#  Copyright 2002-2019 Barcelona Supercomputing Center (www.bsc.es)
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
from pycompss.api.parameter import *
import random
import sys
import time

def getParents(population, target, retain=0.2):
    fitInd = [(p, fitness(p, target)) for p in population]
    sortFitInd = sorted(fitInd, key=lambda i: i[1])
    numRetain = int(len(population)*retain)
    return [sortFitInd[i][0] for i in range(numRetain)]

@task(returns=list)
def mutate(p):
    ind = random.randint(0, len(p)-1)
    p[ind] = random.randint(min(p), max(p))
    return p

@task(returns=list)
def crossover(male, female):
    half = len(male)/2
    child = male[:half] + female[half:]
    return child


@task(returns=list)
def individual(size):
    return [random.randint(0, 100) for _ in range(size)]


def genPopulation(numIndividuals, size):
    return [individual(size) for _ in range(numIndividuals)]

@task(returns=float)
def fitness(individual, target):
    value = sum(individual)
    return abs(target-value)


def grade(population, target):
    values = map(fitness, population, [target for _ in range(len(population))])
    return sum(values)/float(len(population))


def evolve(population, target, retain=0.2, random_select=0.05, mutate_rate=0.01):
    # get parents
    parents = getParents(population, target, retain)

    # add genetic diversity
    for p in population:
        if p not in parents and random_select > random.random():
            parents.append(p)

    # mutate some individuals
    for p in parents:
        if mutate_rate > random.random():
            p = mutate(p)

    # crossover parents to create childrens
    childrens = []
    numParents = len(parents)
    while len(childrens) < len(population)-numParents:
        male = random.randint(0, numParents-1)
        female = random.randint(0, numParents-1)
        if male != female:
            childrens.append(crossover(parents[male], parents[female]))

    newpopulation = parents + childrens
    # return population
    return newpopulation

if __name__ == "__main__":
    from pycompss.api.api import compss_wait_on
    N = int(sys.argv[1]) #100  # individuals
    size = int(sys.argv[2]) #100  # size of individuals
    x = int(sys.argv[3]) #200  # target
    lifeCycles = int(sys.argv[4])  #10

    print ("----PARAMS: \n N: {} \n size: {} \n x: {} \n lifeCycles: {}\n-----------".format(N, size, x, lifeCycles))
    st = time.time()
    p = genPopulation(N, size)
    et = time.time()
    print ("genPopulation: Elapsed Time {} (s)".format(et-st))
    #fitnessHistory = [grade(p, x)]
    for i in range(lifeCycles):
        p = evolve(p, x)
        #fitnessHistory.append(grade(p,x))
    else:
        p = compss_wait_on(p)
        #fitnessHistory = grade(p,x)
        print ("genAlgorithm: Elapsed Time {} (s)".format(time.time()-et))
        #print "final fitness: {}".format(fitnessHistory)

