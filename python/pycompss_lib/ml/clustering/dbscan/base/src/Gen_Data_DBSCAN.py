from pycompss.api.api import compss_wait_on
from pycompss.api.task import task
from ast import literal_eval
import os
import sys
import numpy as np
from itertools import product

@task(returns=list)
def init_data(comb, path, num_points_max, centers, std):
    data = []
    dim = len(comb)
    for i in range(len(centers)):
        for j in range(num_points_max):
            tmp = np.random.multivariate_normal(centers[i], std[i], size=(1))
            for num, k in enumerate(comb):
                if (tmp[0][num] <= float(k)/10 or
                        tmp[0][num] > float(k + 1)/10):
                    break
            else:
                data.append(tmp)
    if len(data) > 0:
        data = np.vstack(data)
        tmp_vec = -2*np.ones(np.shape(data)[0])
#        data = [data, tmp_vec]
        np.savetxt(path, data)
        return [len(tmp_vec)]
    else:
        data = np.random.sample((1, dim))
        for num, k in enumerate(comb):
            data[0][num] = np.random.uniform(low=float(k)/10,
                                             high=float(k+1)/10)
#        tmp_vec = -2*np.ones(np.shape(data)[0])
#        data = [data, tmp_vec]
        np.savetxt(path, data)
        return [1]
#    return data_pos

def main(file_count, dimensions):
    dimensions = literal_eval(dimensions)
    num_points_max = 30
    dim = len(dimensions)
    num_centers = len(dimensions)
    centers = np.random.sample((num_centers, dim))
    std = np.random.sample((num_centers, dim, dim))
    for i, c in enumerate(std):
        std[i] = np.dot(std[i], std[i].transpose())/2
#    path = "/gpfs/projects/bsc19/COMPSs_DATASETS/dbscan/"+str(file_count)
    path = "~/DBSCAN/data/"+str(file_count)
    path = os.path.expanduser(path)
#    if not(os.path.exists(path)):
#        os.makedirs(path)
    perm = []
    prod = 1
    for i in range(len(dimensions)):
        perm.append(range(int(dimensions[i])))
        prod = prod*int(dimensions[i])
    lengths = [[] for _ in range(prod)]
    for i, comb in enumerate(product(*perm)):
        tmp_string = path+"/"+str(comb[0])
        for num, j in enumerate(comb):
            if num > 0:
                tmp_string += "_"+str(j)
        tmp_string += ".txt"
        lengths[i] = init_data(comb, tmp_string, num_points_max, centers, std)
    lengths = compss_wait_on(lengths)
    lengths = [a for m in lengths for a in m]
    print "Number of points created: " + str(sum(lengths))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
