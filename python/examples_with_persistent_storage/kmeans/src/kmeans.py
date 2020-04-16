import time
import numpy as np

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from pycompss.api.api import compss_barrier

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances


@task(returns=np.ndarray)
def partial_sum(fragment, centres):
    partials = np.zeros((centres.shape[0], 2), dtype=object)
    arr = fragment.mat
    close_centres = pairwise_distances(arr, centres).argmin(axis=1)
    for center_idx, _ in enumerate(centres):
        indices = np.argwhere(close_centres == center_idx).flatten()
        partials[center_idx][0] = np.sum(arr[indices], axis=0)
        partials[center_idx][1] = indices.shape[0]
    return partials


@task(returns=dict)
def merge(*data):
    accum = data[0].copy()
    for d in data[1:]:
        accum += d
    return accum


def converged(old_centres, centres, epsilon, iteration, max_iter):
    if old_centres is None:
        return False
    dist = np.sum(paired_distances(centres, old_centres))
    return dist < epsilon ** 2 or iteration >= max_iter


def recompute_centres(partials, old_centres, arity):
    centres = old_centres.copy()
    while len(partials) > 1:
        partials_subset = partials[:arity]
        partials = partials[arity:]
        partials.append(merge(*partials_subset))
    partials = compss_wait_on(partials)
    for idx, sum_ in enumerate(partials[0]):
        if sum_[1] != 0:
            centres[idx] = sum_[0] / sum_[1]
    return centres


def kmeans_frag(fragments, dimensions, num_centres=10, iterations=20,
                seed=0., epsilon=1e-9, arity=50):
    """
    A fragment-based K-Means algorithm.
    Given a set of fragments (which can be either PSCOs or future objects that
    point to PSCOs), the desired number of clusters and the maximum number of
    iterations, compute the optimal centres and the index of the centre
    for each point.
    PSCO.mat must be a NxD float np.ndarray, where D = dimensions
    :param fragments: Number of fragments
    :param dimensions: Number of dimensions
    :param num_centres: Number of centres
    :param iterations: Maximum number of iterations
    :param seed: Random seed
    :param epsilon: Epsilon (convergence distance)
    :param arity: Arity
    :return: Final centres
    """
    # Set the random seed
    np.random.seed(seed)
    # Centres is usually a very small matrix, so it is affordable to have it in
    # the master.
    centres = np.asarray(
        [np.random.random(dimensions) for _ in range(num_centres)]
    )
    # Note: this implementation treats the centres as files, never as PSCOs.
    old_centres = None
    iteration = 0
    while not converged(old_centres, centres, epsilon, iteration, iterations):
        print("Doing iteration #%d/%d" % (iteration + 1, iterations))
        old_centres = centres.copy()
        partials = []
        for frag in fragments:
            partial = partial_sum(frag, old_centres)
            partials.append(partial)
        centres = recompute_centres(partials, old_centres, arity)
        iteration += 1
    return centres


def parse_arguments():
    """
    Parse command line arguments. Make the program generate
    a help message in case of wrong usage.
    :return: Parsed arguments
    """
    import argparse
    parser = argparse.ArgumentParser(description='KMeans Clustering.')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Pseudo-random seed. Default = 0')
    parser.add_argument('-n', '--numpoints', type=int, default=100,
                        help='Number of points. Default = 100')
    parser.add_argument('-d', '--dimensions', type=int, default=2,
                        help='Number of dimensions. Default = 2')
    parser.add_argument('-c', '--num_centres', type=int, default=5,
                        help='Number of centres. Default = 2')
    parser.add_argument('-f', '--fragments', type=int, default=10,
                        help='Number of fragments.' +
                             ' Default = 10. Condition: fragments < points')
    parser.add_argument('-m', '--mode', type=str, default='uniform',
                        choices=['uniform', 'normal'],
                        help='Distribution of points. Default = uniform')
    parser.add_argument('-i', '--iterations', type=int, default=20,
                        help='Maximum number of iterations')
    parser.add_argument('-e', '--epsilon', type=float, default=1e-9,
                        help='Epsilon. Kmeans will stop when:' +
                             ' |old - new| < epsilon.')
    parser.add_argument('-a', '--arity', type=int, default=50,
                        help='Arity of the reduction carried out during \
                        the computation of the new centroids')
    parser.add_argument('--use_storage', action='store_true',
                        help='Use storage?')
    return parser.parse_args()


@task(returns=1)
def generate_fragment(points, dim, mode, seed, use_storage):
    """
    Generate a random fragment of the specified number of points using the
    specified mode and the specified seed. Note that the generation is
    distributed (the master will never see the actual points).
    :param points: Number of points
    :param dim: Number of dimensions
    :param mode: Dataset generation mode
    :param seed: Random seed
    :param use_storage: Boolean use storage
    :return: Dataset fragment
    """
    # from storage_model.fragment import Fragment
    # ret = Fragment('frag' + str(seed))
    # return ret  # If we are using an snapshot thats enough

    # Random generation distributions
    rand = {
        'normal': lambda k: np.random.normal(0, 1, k),
        'uniform': lambda k: np.random.random(k),
    }
    r = rand[mode]
    np.random.seed(seed)
    mat = np.asarray(
        [r(dim) for __ in range(points)]
    )
    # Normalize all points between 0 and 1
    mat -= np.min(mat)
    mx = np.max(mat)
    if mx > 0.0:
        mat /= mx

    # Create a Fragment and persist it in our storage.
    if use_storage:
        from storage_model.fragment import Fragment
        ret = Fragment()
        ret.mat = mat  # Overwrite the object content with the generated matrix
        # The seed is different for each fragment, so I use it as id)
        ret.make_persistent('frag' + str(seed))
    else:
        from model.fragment import Fragment
        ret = Fragment()
        ret.mat = mat
    return ret


def main(seed, numpoints, dimensions, num_centres, fragments, mode, iterations,
         epsilon, arity, use_storage):
    """
    This will be executed if called as main script. Look at the kmeans_frag
    for the KMeans function.
    This code is used for experimental purposes.
    I.e it generates random data from some parameters that determine the size,
    dimensionality and etc and returns the elapsed time.
    :param seed: Random seed
    :param numpoints: Number of points
    :param dimensions: Number of dimensions
    :param num_centres: Number of centres
    :param fragments: Number of fragments
    :param mode: Dataset generation mode
    :param iterations: Number of iterations
    :param epsilon: Epsilon (convergence distance)
    :param arity: Arity
    :param use_storage: Boolean to use storage
    :return: None
    """
    start_time = time.time()

    # Generate the data
    fragment_list = []
    # Prevent infinite loops in case of not-so-smart users
    points_per_fragment = max(1, numpoints // fragments)

    for l in range(0, numpoints, points_per_fragment):
        # Note that the seed is different for each fragment.
        # This is done to avoid having repeated data.
        r = min(numpoints, l + points_per_fragment)

        fragment_list.append(
            generate_fragment(r - l, dimensions, mode, seed + l, use_storage)
        )

    compss_barrier()
    print("Generation/Load done")
    initialization_time = time.time()
    print("Starting kmeans")

    # Run kmeans
    centres = kmeans_frag(fragments=fragment_list,
                          dimensions=dimensions,
                          num_centres=num_centres,
                          iterations=iterations,
                          seed=seed,
                          epsilon=epsilon,
                          arity=arity)
    compss_barrier()
    print("Ending kmeans")
    kmeans_time = time.time()

    # Run again kmeans (system cache will be filled)
    print("Second kmeans")
    centres = kmeans_frag(fragments=fragment_list,
                          dimensions=dimensions,
                          num_centres=num_centres,
                          iterations=iterations,
                          seed=seed,
                          epsilon=epsilon,
                          arity=arity)
    compss_barrier()
    print("Ending second kmeans")
    kmeans_2nd = time.time()

    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
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
    options = parse_arguments()
    main(**vars(options))
