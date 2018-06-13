#Imports
from collections import defaultdict # General Imports
import os
import numpy as np
from pycompss.api.task import task # PyCOMPSs Imports
from classes.Data import Data # DBSCAN Imports

def orquestrate_sync_clusters(data, adj_mat, epsilon, coord, neigh_sq_loc,
                              len_neighs, quocient, res, fut_list, TH_2,
                              count_tasks, *args):
    if (len_neighs/quocient) > TH_2:
        [fut_list,
         count_tasks] = orquestrate_sync_clusters(data, adj_mat, epsilon, coord,
                                                  neigh_sq_loc, len_neighs,
                                                  quocient*2, res*2 + 0, fut_list,
                                                  TH_2, count_tasks, *args)
        [fut_list,
         count_tasks] = orquestrate_sync_clusters(data, adj_mat, epsilon, coord,
                                                  neigh_sq_loc, len_neighs,
                                                  quocient*2, res*2 + 1, fut_list,
                                                  TH_2, count_tasks, *args)
    else:
        count_tasks += 1
        fut_list.append(sync_clusters(data, adj_mat, epsilon, coord,
                                      neigh_sq_loc, quocient, res, len_neighs,
                                      *args))
    return fut_list, count_tasks

@task(returns=list)
def merge_task_sync(adj_mat, *args):
    adj_mat_copy = [[] for _ in range(max(adj_mat[0], 1))]
    for args_i in args:
        for num, list_elem in enumerate(args_i):
            for elem in list_elem:
                if elem not in adj_mat_copy[num]:
                    adj_mat_copy[num].append(elem)
    return adj_mat_copy

@task(returns=list)
def sync_clusters(data, adj_mat, epsilon, coord, neigh_sq_loc, quocient,
                  res, len_neighs, *args):
    # TODO: change *args
#    adj_mat_copy = [deque() for _ in range(max(adj_mat[0], 1))]
    adj_mat_copy = [[] for _ in range(max(adj_mat[0], 1))]
    neigh_data = [np.vstack([i.value[0] for i in args]),
                  np.concatenate([i.value[1] for i in args])]
    neigh_data = [np.vstack([i for num, i in
                             enumerate(neigh_data[0])
                             if ((num % quocient) == res)]),
                  np.array([i for num, i in enumerate(neigh_data[1])
                            if ((num % quocient) == res)])]
    tmp_unwrap = [neigh_sq_loc[i] for i in range(len(neigh_sq_loc))
                  for j in range(len(args[i].value[1]))]
    tmp_unwrap = [i for num, i in enumerate(tmp_unwrap) if
                  ((num % quocient) == res)]
    for num1, point1 in enumerate(data.value[0]):
        current_clust_id = int(data.value[1][num1])
        if current_clust_id > -1:
            tmp_vec = (np.linalg.norm(neigh_data[0]-point1, axis=1) -
                       epsilon) < 0
            for num2, poss_neigh in enumerate(tmp_vec):
                loc2 = tmp_unwrap[num2]
                if poss_neigh:
                    clust_ind = int(neigh_data[1][num2])
                    adj_mat_elem = [loc2, clust_ind]
                    if ((clust_ind > -1)
                        and (adj_mat_elem not in
                            adj_mat_copy[current_clust_id])):
                        adj_mat_copy[current_clust_id].append(adj_mat_elem)
    return adj_mat_copy
