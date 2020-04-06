from __future__ import print_function

import time
import numpy as np

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from pycompss.api.api import compss_barrier


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
def reducecentresTask(a, b):
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
    :param num_centres: Number of centres
    :param iterations: Maximum number of iterations
    :param seed: Random seed
    :param epsilon: Epsilon (convergence distance)
    :param norm: Norm
    :return: Final centres and labels
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
        agg_result = mergeReduce(reducecentresTask, partial_results)
        new_centres, associates, labels = compss_wait_on(agg_result)
        # Normalize
        new_centres /= associates.reshape(len(associates), 1)

        if np.linalg.norm(centres - new_centres, norms[norm]) < epsilon:
            # Convergence criterion is met
            break
        # Convergence criterion is not met, update centres
        centres = new_centres

    return centres, labels


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
    parser.add_argument('-l', '--lnorm', type=str,
                        default='l2', choices=['l1', 'l2'],
                        help='Norm for vectors')
    parser.add_argument('--plot_result', action='store_true',
                        help='Plot the resulting clustering' +
                             ' (only works if dim = 2).')
    parser.add_argument('--use_storage', action='store_true',
                        help='Use storage?')
    return parser.parse_args()


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
    # Create a Fragment and persist it in our storage.
    if use_storage:
        from model.fragment import Fragment
        fragment = Fragment()
        # Make persistent before since it is populated in the task
        fragment.make_persistent()
        fragment.generate_points(points, dim, mode, seed)
    else:
        from model.fake_fragment import Fragment
        fragment = Fragment()
        fragment.generate_points(points, dim, mode, seed)
    return fragment


def plot_result(fragment_list, centres):
    """
    Generate an image showing the points (whose colour determined the cluster
    they belong to) and the centers.
    :param fragment_list: List of fragments
    :param centres: Centres
    :return: None
    """
    import matplotlib.pyplot as plt
    plt.figure('Clustering')

    def color_wheel(i):
        l = ['red', 'purple', 'blue', 'cyan', 'green']
        return l[i % len(l)]

    idx = 0
    for frag in fragment_list:
        frag = compss_wait_on(frag)
        for (i, p) in enumerate(frag.mat):
            col = color_wheel(labels[idx])
            plt.scatter(p[0, 0], p[0, 1], color=col)
            idx += 1
    for centre in centres:
        plt.scatter(centre[0, 0], centre[0, 1], color='black')
    import uuid
    plt.savefig('%s.png' % str(uuid.uuid4()))


def main(seed, numpoints, dimensions, num_centres, fragments, mode, iterations,
         epsilon, lnorm, plot_result, use_storage):
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
    :param lnorm: Norm to use
    :param plot_result: Boolean to plot result
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
    centres, labels = kmeans_frag(fragments=fragment_list,
                                  dimensions=dimensions,
                                  num_centres=num_centres,
                                  iterations=iterations,
                                  seed=seed,
                                  epsilon=epsilon,
                                  norm=lnorm)
    compss_barrier()
    print("Ending kmeans")
    kmeans_time = time.time()

    # Run again kmeans (system cache will be filled)
    print("Second kmeans")
    centres, labels = kmeans_frag(fragments=fragment_list,
                                  dimensions=dimensions,
                                  num_centres=num_centres,
                                  iterations=iterations,
                                  seed=seed,
                                  epsilon=epsilon,
                                  norm=lnorm)
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

    # Plot results if possible
    if dimensions == 2 and plot_result:
        plot_result(fragment_list, centres)


if __name__ == "__main__":
    options = parse_arguments()
    main(**vars(options))
