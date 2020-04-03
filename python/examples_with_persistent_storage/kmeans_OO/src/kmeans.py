from __future__ import print_function

import time
import sys
import numpy as np

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from pycompss.api.api import compss_barrier

from model.fragment import Fragment


#############################################
# Constants / experiment values:
#############################################

# NUMPOINTS = 40000
# FRAGMENTS = 100
# DIMENSIONS = 20
# CENTERS = 20
MODE = 'uniform'
SEED = 42
ITERATIONS = 5


#############################################
#############################################

def mergeReduce(function, data):
    """
    Apply function cumulatively to the items of data,
    from left to right in binary tree structure, so as to
    reduce the data to a single value.
    :param function: function to apply to reduce data
    :param data: List of items to be reduced
    :return: result of reduce the data to a single value
    """
    from collections import deque
    q = deque(list(range(len(data))))
    while len(q):
        x = q.popleft()
        if len(q):
            y = q.popleft()
            data[x] = function(data[x], data[y])
            q.append(x)
        else:
            return data[x]


@task(returns=tuple, priority=True)
def reduceCentersTask(a, b):
    """
    Reduce method to sum the result of two partial_sum methods
    :param a: partial_sum matrix containing the sum and the cardinal
    :param b: partial_sum matrix containing the sum and the cardinal
    :return: the sum a + b
    """
    a_sum, a_associates, a_labels = a
    b_sum, b_associates, b_labels = b

    a_sum += b_sum
    a_associates += b_associates
    a_labels.extend(b_labels)

    return (a_sum, a_associates, a_labels)


def kmeans_frag(fragments, dimensions, num_centres=10, iterations=20,
                seed=0., epsilon=1e-9, norm='l2'):
    """
    A fragment-based K-Means algorithm.
    Given a set of fragments (which can be either PSCOs or future objects that
    point to PSCOs), the desired number of clusters and the maximum number of
    iterations, compute the optimal centres and the index of the centre
    for each point.
    PSCO.mat must be a NxD float np.matrix, where D = dimensions
    :param fragments: Number of fragments
    :param dimensions: Number of dimensions
    :param num_centres: Number of centers
    :param iterations: Maximum number of iterations
    :param seed: Random seed
    :param epsilon: Epsilon (convergence distance)
    :param norm: Norm
    :return: Final centers and labels
    """
    # Choose the norm among the available ones
    norms = {
        'l1': 1,
        'l2': 2,
    }
    # Set the random seed
    np.random.seed(seed)
    # Centres is usually a very small matrix, so it is affordable to have it in
    # the master.
    centres = np.matrix(
        [np.random.random(dimensions) for _ in range(num_centres)]
    )

    # Note: this implementation treats the centres as files, never as PSCOs.
    for it in range(iterations):
        print("Doing iteration #%d/%d" % (it + 1, iterations))
        partial_results = []
        current_labels = []
        for frag in fragments:
            # For each fragment compute, for each point, the nearest centre.
            # Return the mean sum of the coordinates assigned to each centre.
            # Note that mean = mean ( sum of sub-means )
            partial_result = frag.cluster_and_partial_sums(centres,
                                                           norms[norm])
            partial_results.append(partial_result)

        # Aggregate results
        agg_result = mergeReduce(reduceCentersTask, partial_results)
        new_centres, associates, labels = compss_wait_on(agg_result)
        # Normalize
        new_centres /= associates.reshape(len(associates), 1)

        if np.linalg.norm(centres - new_centres, norms[norm]) < epsilon:
            # Convergence criterion is met
            break
        # Convergence criterion is not met, update centres
        centres = new_centres

    return centres, labels


def main():
    NUMPOINTS = int(sys.argv[1])
    FRAGMENTS = int(sys.argv[2])
    DIMENSIONS = int(sys.argv[3])
    CENTERS = int(sys.argv[4])

    start_time = time.time()

    # Generate the data
    fragment_list = []
    # Prevent infinite loops in case of not-so-smart users
    points_per_fragment = NUMPOINTS // FRAGMENTS

    for i, l in enumerate(range(0, NUMPOINTS, points_per_fragment)):
        # Note that the seed is different for each fragment.
        # This is done to avoid having repeated data.
        r = min(NUMPOINTS, l + points_per_fragment)

        fragment = Fragment()
        fragment.make_persistent()
        fragment.generate_points(r - l, DIMENSIONS, MODE, SEED + l)

        fragment_list.append(fragment)

    compss_barrier()
    print("Generation/Load done")
    initialization_time = time.time()
    print("Starting kmeans")

    # Run kmeans
    num_centers = CENTERS
    centres, labels = kmeans_frag(fragments=fragment_list,
                                  dimensions=DIMENSIONS,
                                  num_centres=num_centers,
                                  iterations=ITERATIONS,
                                  seed=SEED)
    compss_barrier()
    print("Ending kmeans")

    kmeans_time = time.time()

    print("Second round of kmeans")
    centres, labels = kmeans_frag(fragments=fragment_list,
                                  dimensions=DIMENSIONS,
                                  num_centres=num_centers,
                                  iterations=ITERATIONS,
                                  seed=SEED)
    compss_barrier()
    print("Ending kmeans")
    kmeans_2nd = time.time()

    print("-----------------------------------------")
    print("----- RESULTS (PyCOMPSs + Storage) ------")
    print("-----------------------------------------")
    print("Initialization time: %f" % (initialization_time - start_time))
    print("Kmeans time: %f" % (kmeans_time - initialization_time))
    print("Kmeans 2nd round time: %f" % (kmeans_2nd - kmeans_time))
    print("Total time: %f" % (kmeans_2nd - start_time))
    print("-----------------------------------------")
    centres = compss_wait_on(centres)
    print("CENTRES:")
    print(centres)
    print("-----------------------------------------")


if __name__ == "__main__":
    main()
