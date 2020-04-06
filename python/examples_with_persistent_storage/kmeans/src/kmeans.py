from __future__ import print_function

import time
import numpy as np

from pycompss.api.task import task
from pycompss.api.parameter import INOUT
from pycompss.api.api import compss_wait_on
from pycompss.api.api import compss_barrier


@task(returns=1, labels=INOUT)
def cluster_and_partial_sums(fragment, labels, centres, norm):
    """
    Given a fragment of points, declare a CxD matrix A and, for each point p:
    1) Compute the nearest centre c of p
    2) Add p / num_points_in_fragment to A[index(c)]
    3) Set label[index(p)] = c
    :param fragment: Fragment
    :param labels: Labels
    :param centres: Centers
    :param norm: Norm for normalization
    :return: Distances to centers of each point
    """
    ret = np.matrix(np.zeros(centres.shape))
    n = fragment.mat.shape[0]
    c = centres.shape[0]
    # Check if labels is an empty list
    if not labels:
        # If it is, fill it with n zeros.
        for _ in range(n):
            # Done this way to not lose the reference
            labels.append(0)
    # Compute the big stuff
    associates = np.zeros(c)
    # Get the labels for each point
    for (i, point) in enumerate(fragment.mat):
        distances = np.zeros(c)
        for (j, centre) in enumerate(centres):
            distances[j] = np.linalg.norm(point - centre, norm)
        labels[i] = np.argmin(distances)
        associates[labels[i]] += 1
    # Add each point to its associate centre
    for (i, point) in enumerate(fragment.mat):
        ret[labels[i]] += point / associates[labels[i]]
    return ret


def kmeans_frag(fragments, dimensions, num_centres=10, iterations=20,
                seed=0., epsilon=1e-9, norm='l2'):
    """
    A fragment-based K-Means algorithm.
    Given a set of fragments (which can be either Fragments or future objects
    that point to Fragments), the desired number of clusters and the maximum
    number of iterations, compute the optimal centres and the index of the
    centre for each point.
    Fragment.mat must be a NxD float np.matrix, where D = dimensions
    :param fragments: Number of fragments
    :param dimensions: Number of dimensions
    :param num_centres: Number of centers
    :param iterations: Maximum number of iterations
    :param seed: Random seed
    :param epsilon: Epsilon (convergence distance)
    :param norm: Norm
    :return: Final centers and labels
    """
    import numpy as np
    # Choose the norm among the available ones
    norms = {
        'l1': 1,
        'l2': 2
    }
    # Set the random seed
    np.random.seed(seed)
    # Centres is usually a very small matrix, so it is affordable to have it in
    # the master.
    centres = np.matrix(
        [np.random.random(dimensions) for _ in range(num_centres)]
    )
    # Make a list of labels, treat it as INOUT
    # Leave it empty at the beginning, update it inside the task. Avoid
    # having a linear amount of stuff in master's memory unnecessarily
    labels = [[] for _ in range(len(fragments))]
    # Note: this implementation treats the centres as files, never as Fragments
    for it in range(iterations):
        partial_results = []
        for (i, frag) in enumerate(fragments):
            # For each fragment compute, for each point, the nearest centre.
            # Return the mean sum of the coordinates assigned to each centre.
            # Note that mean = mean ( sum of sub-means )
            partial_result = cluster_and_partial_sums(frag, labels[i],
                                                      centres, norms[norm])
            partial_results.append(partial_result)
        # Bring the partial sums to the master and
        # compute new centres when syncing
        new_centres = np.matrix(np.zeros(centres.shape))
        from pycompss.api.api import compss_wait_on
        for partial in partial_results:
            partial = compss_wait_on(partial)
            # Mean of means, single step
            new_centres += partial / float(len(fragments))
        if np.linalg.norm(centres - new_centres, norms[norm]) < epsilon:
            # Convergence criterion is met
            break
        # Convergence criterion is not met, update centres
        centres = new_centres
    # If we are here either we have converged or we have run out of iterations
    # In any case, now it is time to update the labels in the master
    ret_labels = []
    for label_list in labels:
        from pycompss.api.api import compss_wait_on
        to_add = compss_wait_on(label_list)
        ret_labels += to_add
    return centres, ret_labels


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
    parser.add_argument('-c', '--centres', type=int, default=5,
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
    # from model.fragment import Fragment
    # ret = Fragment('frag' + str(seed))
    # return ret  # If we are using an snapshot thats enough

    import numpy as np

    # Random generation distributions
    def normal(k):
        return np.random.normal(0, 1, k)

    def uniform(k):
        return np.random.random(k)

    rand = {
        'normal': normal,
        'uniform': uniform
    }
    r = rand[mode]
    np.random.seed(seed)
    mat = np.matrix(
        [r(dim) for __ in range(points)]
    )
    # Normalize all points between 0 and 1
    mat -= np.min(mat)
    mx = np.max(mat)
    if mx > 0.0:
        mat /= mx

    # Create a Fragment and persist it in our storage.
    if use_storage:
        from model.fragment import Fragment
        ret = Fragment()
        ret.mat = mat  # Overwrite the object content with the generated matrix
        # The seed is different for each fragment, so I use it as id)
        ret.make_persistent('frag' + str(seed))
    else:
        from model.fake_fragment import Fragment
        ret = Fragment()
        ret.mat = mat
    return ret


def main(seed, numpoints, dimensions, centres, fragments, mode, iterations,
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
    :param centres: Number of centers
    :param fragments: Number of fragments
    :param mode: Dataset generation mode
    :param iterations: Number of iterations
    :param epsilon: Epsilon (convergence distance)
    :param lnorm: Norm to use
    :param plot_result: Boolean to plot result
    :param use_storage: Boolean to use storage
    :return: None
    """
    from pycompss.api.api import compss_barrier
    import time

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

    initialization_time = time.time()

    # Run kmeans
    centres, labels = kmeans_frag(fragments=fragment_list,
                                  dimensions=dimensions,
                                  num_centres=centres,
                                  iterations=iterations,
                                  seed=seed,
                                  epsilon=epsilon,
                                  norm=lnorm)

    compss_barrier()
    kmeans_time = time.time()

    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Initialization time: %f" % (initialization_time - start_time))
    print("Kmeans time: %f" % (kmeans_time - initialization_time))
    print("Total time: %f" % (kmeans_time - start_time))
    print("Centres: " + str(centres))
    print("-----------------------------------------")

    # Plot results if possible
    if dimensions == 2 and plot_result:
        import matplotlib.pyplot as plt
        plt.figure('Clustering')

        def color_wheel(i):
            l = ['red', 'purple', 'blue', 'cyan', 'green']
            return l[i % len(l)]

        idx = 0
        for frag in fragment_list:
            from pycompss.api.api import compss_wait_on
            frag = compss_wait_on(frag)
            for (i, p) in enumerate(frag.mat):
                col = color_wheel(labels[idx])
                plt.scatter(p[0, 0], p[0, 1], color=col)
                idx += 1
        for centre in centres:
            plt.scatter(centre[0, 0], centre[0, 1], color='black')
        import uuid
        plt.savefig('%s.png' % str(uuid.uuid4()))


if __name__ == "__main__":
    options = parse_arguments()
    main(**vars(options))
