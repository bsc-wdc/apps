#Imports
from collections import defaultdict # General Imports
import numpy as np
from pycompss.api.task import task # PyCOMPSs Imports
from classes.Data import Data # DBSCAN Imports

def orquestrate_scan_merge(data, epsilon, min_points, len_neighs, quocient,
                           res, fut_list, TH_1, count_tasks, *args):
    if (len_neighs/quocient) > TH_1:
        [fut_list[0],
         fut_list[1],
         count_tasks] = orquestrate_scan_merge(data, epsilon, min_points,
                                               len_neighs, quocient*2, res*2 + 0,
                                               fut_list, TH_1, count_tasks,
                                               *args)
        [fut_list[0],
         fut_list[1],
         count_tasks] = orquestrate_scan_merge(data, epsilon, min_points,
                                               len_neighs, quocient*2, res*2 + 1,
                                               fut_list, TH_1, count_tasks,
                                               *args)
    else:
        obj = [[], []]
        count_tasks += 1
        obj[0], obj[1] = partial_scan_merge(data, epsilon, min_points,
                                            quocient, res, *args)
        for num, _list in enumerate(fut_list):
            _list.append(obj[num])
    return fut_list[0], fut_list[1], count_tasks

@task(returns=2)
def partial_scan_merge(data, epsilon, min_points, quocient, res, *args):
    # Core point location in the data chunk
    data_copy = Data()
    data_copy.value = [np.array([i for num, i in enumerate(data.value[0])
                                 if ((num % quocient) == res)]),
                       np.array([i for num, i in enumerate(data.value[1])
                                 if ((num % quocient) == res)])]
    neigh_points = np.vstack([i.value[0] for i in args])
    neigh_points_clust = np.concatenate([i.value[1] for i in args])
    non_assigned = defaultdict(int)
    for num, point in enumerate(data_copy.value[0]):
        poss_neigh = np.linalg.norm(neigh_points-point, axis=1) - epsilon < 0
        neigh_count = np.sum(poss_neigh)
        if neigh_count > min_points:
            data_copy.value[1][num] = -1
        elif neigh_count > 0:
            tmp = []
            for pos, proxim in enumerate(poss_neigh):
                if proxim:
                    # Adding three since later to detect core points we will
                    # require value > -1 and -0 is > 1 and not to confuse it
                    # with a noise point
                    if neigh_points_clust[pos] == -1:
                        data_copy.value[1][num] = -(pos+3)
                        break
                    else:
                        tmp.append(pos)
            else:
                non_assigned[num] = tmp

    # Local clustering
    cluster_count = 0
    core_points_tmp = [p for num, p in enumerate(data_copy.value[0])
                                 if data_copy.value[1][num] == -1]
    if len(core_points_tmp) != 0:
        core_points_tmp = np.vstack(core_points_tmp)
        core_points = [[num, p] for num, p in enumerate(core_points_tmp)]
        for pos, point in core_points:
            tmp_vec = ((np.linalg.norm(core_points_tmp - point, axis=1)-epsilon)
                       < 0)
            for num, poss_neigh in enumerate(tmp_vec):
                if poss_neigh and data_copy.value[1][num] > -1:
                    data_copy.value[1][pos] = data_copy.value[1][num]
                    break
            else:
                data_copy.value[1][pos] = cluster_count
                cluster_count += 1
    return data_copy, non_assigned

@task(returns=3)
def merge_task_ps_0(*args):
    num_clust_max = max(max([i for i in args[0].value[1]])+1,0)
    data = args[0]
    for i in range(len(args)-1):
        data.value = [np.vstack([data.value[0], args[i+1].value[0]]),
                      np.concatenate([data.value[1], [j + num_clust_max if
                                                      j > -1 else j for j in
                                                      args[i+1].value[1]]])]
        num_clust_max += max(max([j for j in args[i+1].value[1]])+1, 0)
    return data, [int(num_clust_max)], [int(num_clust_max)]

@task(returns=defaultdict)
def merge_task_ps_1(*args):
    # This one is for data type
    border_points = defaultdict(list)
    for _dict in args:
        for key in _dict:
            border_points[key] += _dict[key]
    return border_points
