#Imports
    # General Imports
from collections import defaultdict
import os
import itertools
import copy
import numpy as np
    # PyCOMPSs Imports
from pycompss.api.task import task
    # DBSCAN Imports
from classes.Data import Data
from classes.DS import DisjointSet


def unwrap_adj_mat(tmp_mat, adj_mat, neigh_sq_coord, dimension_perms):
    links_list = []
    for comb in itertools.product(*dimension_perms):
        for k in range(tmp_mat[comb][0]):
            links_list.append([comb, k])
    mf_set = DisjointSet(links_list)
    for comb in itertools.product(*dimension_perms):
            # For an implementation issue, elements in adj_mat are
            # wrapped with an extra pair of [], hence the j[0]
        for k in range(len(adj_mat[comb][0])-1):
            mf_set.union(adj_mat[comb][0][k], adj_mat[comb][0][k+1])
    out = mf_set.get()
    return out

@task()
def expand_cluster(data, epsilon, border_points, dimension_perms, links_list,
                   square, cardinal, file_id, is_mn, *args):
    data_copy = Data()
    data_copy.value = copy.deepcopy(data.value)
    tmp_unwrap_2 = [i.value[1] for i in args]
    neigh_points_clust = np.concatenate(tmp_unwrap_2)
    for elem in border_points:
        for p in border_points[elem]:
            if neigh_points_clust[p] > -1:
                data_copy.value[1][elem] = neigh_points_clust[p]
                break
    for num, elem in enumerate(data_copy.value[1]):
        if elem < -2:
            clust_ind = -1*elem - 3
            data_copy.value[1][num] = neigh_points_clust[clust_ind]

    # Map all cluster labels to general numbering
    mappings = []
    for k, links in enumerate(links_list):
        for pair in links:
            if pair[0] == square:
                mappings.append([pair[1], k])
    for num, elem in enumerate(data_copy.value[1]):
        if elem > -1:
            for pair in mappings:
                if int(elem) == pair[0]:
                    data_copy.value[1][num] = pair[1]

    # Update all files (for the moment writing to another one)
    if is_mn:
        path = "/gpfs/projects/bsc19/COMPSs_DATASETS/dbscan/"+str(file_id)
    else:
        path = "~/DBSCAN/data/"+str(file_id)
    path = os.path.expanduser(path)
    tmp_string = path+"/"+str(square[0])
    for num, j in enumerate(square):
        if num > 0:
            tmp_string += "_"+str(j)
    tmp_string += "_OUT.txt"
    f_out = open(tmp_string, "w")
    for num, val in enumerate(data_copy.value[0]):
        f_out.write(str(data_copy.value[0][num])+" "
                    + str(int(data_copy.value[1][num])) + "\n")
    f_out.close()

