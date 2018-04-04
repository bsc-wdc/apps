#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#  Copyright 2002-2015 Barcelona Supercomputing Center (www.bsc.es)
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
import numpy as np
from pycompss.api.task import task


class Fragment(object):
    def __init__(self, dim, points, base_index):
        self.dim = dim
        self.points = points
        self.base_index = base_index

    @task(returns=dict)
    def cluster_points(self, mu):
        base_idx = self.base_index
        dic = dict()

        for x in enumerate(self.points):
            bestmukey = min([(i[0], np.linalg.norm(x[1] - mu[i[0]]))
                             for i in enumerate(mu)], key=lambda t: t[1])[0]
            if bestmukey not in dic:
                dic[bestmukey] = [x[0] + base_idx]
            else:
                dic[bestmukey].append(x[0] + base_idx)
        return dic

    @task(returns=dict)
    def partial_sum(self, clusters):
        base_idx = self.base_index
        points = self.points
        dic = {}
        for i in clusters:
            p_idx = np.array(clusters[i]) - base_idx
            dic[i] = (len(p_idx), np.sum(points[p_idx], axis=0))
        return dic

    def __getitem__(self, idx):
        """This class expects `idx` to be a list or an array of indexes.
        Otherwise, performance will suffer *a lot*."""
        return self.points[idx]


class Board(object):
    def __init__(self, dim, points_per_fragment):
        self.dim = dim
        self.fragments = list()
        self.n_points = 0

        # Model programmer stuff, not application stuff, not sure where to put this
        self.points_per_fragment = points_per_fragment

    def init_random(self, n_points, base_seed):
        self.n_points = n_points

        i = 0  # Corner-case: single non-full fragment
        for i in range(n_points // self.points_per_fragment):
            np.random.seed(base_seed + i)
            fragment = Fragment(
                dim=self.dim,
                # Misc note about entropy, seeds and PyCOMPSs implementation: #
                ###############################################################
                # PyCOMPSs initializes with the seed and then iterates all the
                # points_per_fragment-like variable in order to build the
                # fragment. This results in several calls to np.random.random
                # (but all of them are "vector" calls).
                #
                # Here I propose a single "matrix" call, resulting in a
                # different seed per call of random.random. It seems that the
                # entropy usage of numpy is consistent and the preliminary
                # tests indicate that the implementation is equivalent
                # (although I am not sure that this is a spec on the
                # numpy.random behaviour).
                points=np.random.random([self.points_per_fragment, self.dim]),
                base_index=i * self.points_per_fragment
            )
            # fragment.make_persistent(dest_stloc_id=storage_locations.next())
            self.fragments.append(fragment)

        remain = n_points % self.points_per_fragment
        if remain:
            np.random.seed(base_seed + i + 1)
            fragment = Fragment(
                dim=self.dim,
                points=np.random.random([remain, self.dim]),
                base_index=i * self.points_per_fragment
            )
            # fragment.make_persistent(dest_stloc_id=storage_locations.next())
            self.fragments.append(fragment)
