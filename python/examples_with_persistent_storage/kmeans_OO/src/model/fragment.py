try:
    # dataClay
    from storage.api import StorageObject
except:
    try:
        # Hecuba
        from hecuba.storageobj import StorageObj as StorageObject
    except:
        # Redis
        from storage.storage_object import StorageObject


try:
    from pycompss.api.task import task
    from pycompss.api.parameter import IN, INOUT
except ImportError:
    # Required since the pycompss module is not ready during the registry
    from dataclay.contrib.dummy_pycompss import task, IN, INOUT

try:
    from dataclay import dclayMethod
except ImportError:
    def dclayMethod(*args, **kwargs):
        return lambda f: f

import numpy as np


class Fragment(StorageObject):
    """
    @ClassField points numpy.ndarray

    @dclayImport numpy as np
    """
    @dclayMethod()
    def __init__(self):
        super(Fragment, self).__init__()
        self.points = None

    @task(target_direction=INOUT)
    @dclayMethod(num_points='int', dim='int', mode='str', seed='int')
    def generate_points(self, num_points, dim, mode, seed):
        """
        Generate a random fragment of the specified number of points using the
        specified mode and the specified seed. Note that the generation is
        distributed (the master will never see the actual points).
        :param num_points: Number of points
        :param dim: Number of dimensions
        :param mode: Dataset generation mode
        :param seed: Random seed
        :return: Dataset fragment
        """
        # Random generation distributions
        rand = {
            'normal': lambda k: np.random.normal(0, 1, k),
            'uniform': lambda k: np.random.random(k),
        }
        r = rand[mode]
        np.random.seed(seed)
        mat = np.asarray(
            [r(dim) for __ in range(num_points)]
        )
        # Normalize all points between 0 and 1
        mat -= np.min(mat)
        mx = np.max(mat)
        if mx > 0.0:
            mat /= mx

        self.points = mat

    @task(returns=tuple, target_direction=IN)
    @dclayMethod(centres='numpy.matrix', norm='anything', return_='anything')
    def cluster_and_partial_sums(self, centres, norm):
        """
        Given self (fragment == set of points), declare a CxD matrix A and,
        for each point p:
           1) Compute the nearest centre c of p
           2) Add p / num_points_in_fragment to A[index(c)]
           3) Set label[index(p)] = c
        :param centres: Centers
        :param norm: Norm for normalization
        :return: Sum of points for each center, qty of associations for each
                 center, and label for each point
        """
        mat = self.points
        ret = np.zeros(centres.shape)
        n = mat.shape[0]
        c = centres.shape[0]
        labels = list()

        # Compute the big stuff
        associates = np.zeros(c, dtype=int)
        # Get the labels for each point
        for point in mat:
            distances = np.zeros(c)
            for (j, centre) in enumerate(centres):
                distances[j] = np.linalg.norm(point - centre, norm)

            ass = np.argmin(distances)
            labels.append(ass)
            associates[ass] += 1

        # Add each point to its associate centre
        for (label_i, point) in zip(labels, mat):
            ret[label_i] += point
        return (ret, associates, labels)
