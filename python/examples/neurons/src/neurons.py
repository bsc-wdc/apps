#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Code for:
# Chapter 20: Practically Trivial Parallel Data Processing in a Neuroscience Laboratory
# M. Denker, B. Wiebelt, D. Fliegner, M. Diesmann, A. Morrison
# In: Analaysis of Parallel Spike Trains (2010) S. Gruen and S. Rotter (eds). Springer Series in Computational Neuroscience 7
# http://www.spiketrain-analysis.org
#
# Complements section 20.6: Using parallel python on remote nodes running a parallel python server.
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

import pickle
import time
from numpy import *

#load experiment and analysis parameters
from experiment_params import num_neurons,num_secs,num_bins
from analysis_params import maxlag,num_surrs

from pycompss.api.task import task
from pycompss.api.parameter import *


@task(cc_original = INOUT, cc_surrs = INOUT, priority = True)
def gather(result, cc_original, cc_surrs, start, end):
    print("In GATHER task")
    print("GATHER parameters")
    print("- Result[0]:", result[0])
    print("- Result[1]:", result[1])
    print("- cc_original:", cc_original)
    print("- cc_surrs:", cc_surrs)
    print("- cc_original[start:end,:]:", cc_original[start:end,:])
    print("- cc_surrs[start:end,:,:]:", cc_surrs[start:end,:,:])
    print("- start:", start)
    print("- end:", end)
    cc_original[start:end,:] = result[0]
    cc_surrs[start:end,:,:] = result[1]
    print("End of GATHER task")

#cc_surrogate range calculates cc and surrogate cc for a given range of indices
@task(fspikes = FILE, returns = list)
def cc_surrogate_range(fspikes, start_idx, end_idx, seed, num_neurons, num_surrs, num_bins, maxlag):
    print("In CC_SURROGATE_RANGE task")
    print("CC_SURROGATE_RANGE parameters")
    print("- fspikes:", fspikes)
    print("- start:", start_idx)
    print("- end:", end_idx)
    print("- seed:", seed)
    print("- num_neurons:", num_neurons)
    print("- num_surrs:", num_surrs)
    print("- num_bins:", num_bins)
    print("- maxlag:", maxlag)
    random.seed(seed)
    f = open(fspikes, 'r')
    spikes = pickle.load(f)
    idx = 0
    row = 0
    my_cc_original = zeros((end_idx-start_idx,2*maxlag+1))
    my_cc_surrs = zeros((end_idx-start_idx,2*maxlag+1,2))
    idxrange = range(num_bins-maxlag,num_bins+maxlag+1)
    surrs_ij = zeros((num_surrs,2*maxlag+1))
    for ni in arange(num_neurons-1):
        for nj in range(ni+1,num_neurons):
            #get to first index of relevant range
            if (idx < start_idx):
                idx = idx + 1
                continue
            #calculate cc and surrogate ccs for all indices in relevant range
            elif (idx < end_idx):
                print("Calculating correlation between " + str(ni) + " and " + str(nj))
                my_cc_original[row,:] = correlate(spikes[ni,:],spikes[nj,:],"full")[idxrange]
                num_spikes_i = sum(spikes[ni,:])
                num_spikes_j = sum(spikes[nj,:])
                for surrogate in range(num_surrs):
                    print("Calculating surrogate " + str(surrogate))
                    surr_i = zeros(num_bins)
                    surr_i[random.random_integers(0, int(num_bins-1),int(num_spikes_i))] = 1
                    surr_j = zeros(num_bins)
                    surr_j[random.random_integers(0, int(num_bins-1),int(num_spikes_j))] = 1
                    surrs_ij[surrogate,:] = correlate(surr_i,surr_j,"full")[idxrange]
                print("All surrogates done")
                #save point-wise 5% and 95% values of sorted surrogate ccs
                surrs_ij_sorted = sort(surrs_ij,axis=0)
                my_cc_surrs[row,:,0] = surrs_ij_sorted[int(round(num_surrs*0.95)),:]
                my_cc_surrs[row,:,1] = surrs_ij_sorted[int(round(num_surrs*0.05)),:]
                idx = idx + 1
                row = row + 1
                print("End of relevant range")
            #reached end of relevant range; return results
            else:
                print("- cc_original[start:end,:]:", my_cc_original)
                print("- cc_surrs[start:end,:,:]:", my_cc_surrs)
                print("End of CC_SURROGATE_RANGE task")
                return [my_cc_original, my_cc_surrs]

    print("- cc_original[start:end,:]:", my_cc_original)
    print("- cc_surrs[start:end,:,:]:", my_cc_surrs)
    print("End of CC_SURROGATE_RANGE task")
    return [my_cc_original, my_cc_surrs]


if __name__ == "__main__":
    import sys
    from pycompss.api.api import compss_wait_on

    num_frags = int(sys.argv[1])
    fspikes = sys.argv[2]

    num_ccs = int((num_neurons**2 - num_neurons)/2)
    step = int(num_ccs / num_frags)
    remainder = num_ccs % num_frags
    print("Step", step, ", remainder", remainder)
    print("Num_ccs", num_ccs, ", num_frags", num_frags)
    start_idx = 0
    end_idx = 0

    seed = 2398645
    delta = 1782324

    start = time.time()
    #send out tasks
    cc_original = zeros((num_ccs,2*maxlag+1))
    cc_surrs = zeros((num_ccs,2*maxlag+1,2))
    for frag in range(num_frags):
        start_idx = end_idx
        end_idx = end_idx + step
        if remainder > 0:
            end_idx += 1
            remainder -= 1
        #print(start_idx, " -> ", end_idx - 1)
        #print("Got", (end_idx - start_idx), "ccs")
        result = cc_surrogate_range(fspikes, start_idx, end_idx, seed, num_neurons, num_surrs, num_bins, maxlag)
        gather(result, cc_original, cc_surrs, start_idx, end_idx)
        seed = seed + delta
        if frag == 8:
            break

    print("submitted all tasks")

    #save results
    f = open('./result_cc_originals.dat','w')
    cc_original = compss_wait_on(cc_original)

    end = time.time()
    print("Elapsed time(s)")
    print(end - start)

    #print("Originals(", start_idx, "-", end_idx, "):", cc_original[start_idx:end_idx,:]))
    pickle.dump(cc_original,f)
    f.close()

    f = open('./result_cc_surrogates_conf.dat','w')
    cc_surrs = compss_wait_on(cc_surrs)
    #print("Surrogates(", start_idx, "-", end_idx, "):", cc_surrs[start_idx:end_idx,:,:]))
    pickle.dump(cc_surrs,f)
    f.close()

    #end = time.time()
    #print("Elapsed time(s)")
    #print(end - start)
