#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Code for:
# Chapter 20: Practically Trivial Parallel Data Processing in a Neuroscience Laboratory
# M. Denker, B. Wiebelt, D. Fliegner, M. Diesmann, A. Morrison
# In: Analaysis of Parallel Spike Trains (2010) S. Gruen and S. Rotter (eds). Springer Series in Computational Neuroscience 7
# http://www.spiketrain-analysis.org
#
# Listing 20.1
#
# Copyright 2010 Michael Denker, B. Wiebelt, D. Fliegner, M. Diesmann, A. Morrison
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#

from numpy import *
import cPickle as pickle
from pycompss.api.task import task
from pycompss.api.parameter import *

# Analysis params
maxlag = 200
num_surrs = 20  # 1000

# Experiment Params
num_neurons = 100
num_secs = 10
num_bins = num_secs * 1000


@task(cc_original=INOUT, cc_surrs=INOUT, priority=True)
def gather(result, cc_original, cc_surrs, row):
    cc_original[row, :] = result[0]
    cc_surrs[row, :, :] = result[1]


@task(returns=list)
def cc_surrogate(spike_i, spike_j, seed):
    random.seed(seed)
    my_cc_orig = zeros((2 * maxlag + 1))
    my_cc_surrs = zeros((2 * maxlag + 1, num_surrs))
    idxrange = range(num_bins - maxlag, num_bins + maxlag + 1)

    my_cc_orig = correlate(spike_i, spike_j, "full")[idxrange]

    num_spikes_i = sum(spike_i)
    num_spikes_j = sum(spike_j)
    for surrogate in range(num_surrs):
        surr_i = zeros(num_bins)
        surr_i[random.random_integers(0, num_bins - 1, num_spikes_i)] = 1
        surr_j = zeros(num_bins)
        surr_j[random.random_integers(0, num_bins - 1, num_spikes_j)] = 1
        my_cc_surrs[:, surrogate] = correlate(surr_i, surr_j, "full")[idxrange]
    return [my_cc_orig, my_cc_surrs]


if __name__ == '__main__':
    import sys

    # read spike data
    fspikes = sys.argv[1]
    f = open(fspikes, 'r')
    spikes = pickle.load(f)
    f.close()

    # pre-allocate result variables
    num_ccs = (num_neurons**2 - num_neurons) / 2
    cc_orig = zeros((num_ccs, 2 * maxlag + 1))
    cc_surrs = zeros((num_ccs, 2 * maxlag + 1, num_surrs))  # memory error

    row = 0
    seed = 2398645
    delta = 1782324

    # for all pairs ni,nj such that nj > ni
    for ni in range(num_neurons - 1):
        for nj in range(ni + 1, num_neurons):
            print (ni, nj)
            spike_i = spikes[ni, :]
            spike_j = spikes[nj, :]

            result = cc_surrogate(spike_i, spike_j, seed)
            gather(result, cc_orig, cc_surrs, row)
            row += 1
            seed += delta

    # wait and save results
    from pycompss.api.api import compss_wait_on
    cc_orig = compss_wait_on(cc_orig)
    cc_surrs = compss_wait_on(cc_surrs)
    f = open('result_cc_originals.dat', 'w')
    pickle.dump(cc_orig, f)
    f.close()
    f = open('result_cc_surrogates.dat', 'w')
    pickle.dump(cc_surrs, f)
    f.close()
