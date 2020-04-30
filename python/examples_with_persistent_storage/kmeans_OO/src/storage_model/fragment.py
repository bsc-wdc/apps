try:
    # dataClay and Redis
    from storage.api import StorageObject
except:
    # Hecuba
    from hecuba.storageobj import StorageObj as StorageObject

try:
    from pycompss.api.task import task
    from pycompss.api.parameter import IN
except ImportError:
    # Required since the pycompss module is not ready during the registry
    from dataclay.contrib.dummy_pycompss import task, IN

try:
    from dataclay import dclayMethod
except ImportError:
    def dclayMethod(*args, **kwargs):
        return lambda f: f

import numpy as np
from sklearn.metrics import pairwise_distances


class Fragment(StorageObject):
    """
    @ClassField points numpy.ndarray

    @dclayImport numpy as np
    @dclayImportFrom sklearn.metrics import pairwise_distances
    """
    @dclayMethod()
    def __init__(self):
        super(Fragment, self).__init__()
        self.points = None

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

    @task(returns=np.ndarray, target_direction=IN)
    @dclayMethod(centres='numpy.ndarray', return_='anything')
    def partial_sum(self, centres):
        partials = np.zeros((centres.shape[0], 2), dtype=object)
        arr = self.points
        close_centres = pairwise_distances(arr, centres).argmin(axis=1)
        for center_idx, _ in enumerate(centres):
            indices = np.argwhere(close_centres == center_idx).flatten()
            partials[center_idx][0] = np.sum(arr[indices], axis=0)
            partials[center_idx][1] = indices.shape[0]
        return partials
