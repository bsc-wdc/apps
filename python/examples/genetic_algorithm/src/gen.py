#!/usr/bin/python
#
#  Copyright 2002-2022 Barcelona Supercomputing Center (www.bsc.es)
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

import random
import sys
import time
from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.api.api import compss_wait_on


def getParents(population, target, retain=0.2):
    fitInd = [(p, fitness(p, target)) for p in population]
    sortFitIndices(fitInd)
    numRetain = int(len(population) * retain)
    return [fitInd[i][0] for i in range(numRetain)]


@task(fitInd=COLLECTION_INOUT)
def sortFitIndices(fitInd):
    sortFitInd = sorted(fitInd, key=lambda i: i[1])
    for i in range(len(fitInd)):
        fitInd[i] = sortFitInd[i]


@task(returns=list)
def mutate(p, seed):
    random.seed(seed)
    ind = random.randint(0, len(p) - 1)
    p[ind] = random.randint(min(p), max(p))
    return p


@task(returns=list)
def crossover(male, female):
    half = int(len(male) / 2)
    child = male[:half] + female[half:]
    return child


@task(returns=list)
def individual(size, seed):
    random.seed(seed)
    return [random.randint(0, 100) for _ in range(size)]


def genPopulation(numIndividuals, size, seed):
    return [individual(size, seed + i) for i in range(numIndividuals)]


@task(returns=float)
def fitness(individual, target):
    value = sum(individual)
    return abs(target - value)


@task(returns=1, population=COLLECTION_IN)
def grade(population, target):
    values = map(fitness, population, [target for _ in range(len(population))])
    return sum(values) / float(len(population))


def evolve(population, target, seed, retain=0.2, random_select=0.05, mutate_rate=0.01):
    # Get parents
    parents = getParents(population, target, retain)

    # Add genetic diversity
    for p in population:
        if p not in parents and random_select > random.random():
            parents.append(p)

    # Mutate some individuals
    for p in parents:
        if mutate_rate > random.random():
            p = mutate(p, seed)
            seed += 1
    random.seed(seed)

    # Crossover parents to create childrens
    childrens = []
    numParents = len(parents)
    while len(childrens) < len(population) - numParents:
        male = random.randint(0, numParents - 1)
        female = random.randint(0, numParents - 1)
        if male != female:
            childrens.append(crossover(parents[male], parents[female]))

    newpopulation = parents + childrens
    # Return population
    return newpopulation


def main():
    # Input parameters
    N = int(sys.argv[1])  # 100  # individuals
    size = int(sys.argv[2])  # 100  # size of individuals
    x = int(sys.argv[3])  # 200  # target
    lifeCycles = int(sys.argv[4])  # 10
    get_fitness = sys.argv[5] == "True"  # True or False

    seed = 1234

    print("----- PARAMS -----")
    print(f" - N: {N}")
    print(f" - size: {size}")
    print(f" - x: {x}")
    print(f" - lifeCycles: {lifeCycles}")
    print("------------------")

    st = time.time()
    p = genPopulation(N, size, seed)
    et = time.time()
    print("genPopulation: Elapsed Time {} (s)".format(et - st))
    if get_fitness:
        fitnessHistory = [grade(p, x)]
    for i in range(lifeCycles):
        p = evolve(p, x, seed)
        seed += 1
        if get_fitness:
            fitnessHistory.append(grade(p, x))
    else:
        p = compss_wait_on(p)
        print("genAlgorithm: Elapsed Time {} (s)".format(time.time() - et))
        print("Final result: %s" % str(p))
        if get_fitness:
            fitnessHistory.append(grade(p, x))
            fitnessHistory = compss_wait_on(fitnessHistory)
            print("final fitness: {}".format(fitnessHistory))


if __name__ == "__main__":
    main()
