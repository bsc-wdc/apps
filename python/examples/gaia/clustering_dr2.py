import argparse
import os
import resource
from itertools import chain

import numpy as np
import pandas as pd
from pycompss.api.api import compss_barrier
from pycompss.api.task import task

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
import time

header = ['l', 'b', 'par', 'pmra', 'pmdec']

lMinim = 0.
lMaxim = 360.
bMinim = -20.
bMaxim = 20.

box_sizes = [12, 13, 14, 15, 16]
# box_sizes.reverse()
min_samples = [5, 6, 7, 8, 9]
# min_samples.reverse()

tol = 0.2

s_time = time.time()

# In case we want to do a hierarchical model (1 means apply clustering once)
repetitions = 1

n_simulations = 30

count = 0

MAX_BOXES = np.inf


def _do_clustering(in_file, out_dir):
    params = [(L, min_pts) for L in box_sizes for min_pts in min_samples]
    db_results_by_param = []
    for L, min_pts in params:
        stats_file = os.path.join(out_dir, 'stats_L{}_minPts{}.csv'.format(L, min_pts))
        db_results = _compute_boxes(in_file, stats_file, L, min_pts)
        db_results_by_param.append(db_results)
        if count >= MAX_BOXES:
            break
    
    print("Boxes Creation Time: ", "{0:.2f}".format(time.time() - s_time))
    for (db_results, (L, min_pts)) in zip(db_results_by_param, params):
        db_results = db_results_by_param.pop(0)
        
        compss_barrier()
        _postprocess_results(out_dir, L, min_pts, *chain(*db_results))

    compss_barrier()
    print("Main Execution Time: ", "{0:.2f}".format(time.time() - s_time))


@task()
def _postprocess_results(out_dir, L, min_pts, *db_results):
    # Work with data at every box
    columns = header + ['cluster_label', 'cluster_core']
    all_clusters = pd.DataFrame(columns=columns, dtype=float)
    n_label = 0

    for db_res in zip(*[iter(db_results)] * 12):
        shift, i, j, lMin, Ll, bMin, Lb = db_res[0:7]
        df_box, dens, epsilon, n_clusters, labels = db_res[7:12]

        for c in range(n_clusters):
            # find members and cores
            members = _get_members(df_box, c, labels)

            # check if there are members in the edge
            edge = _in_edge(df_box, members, lMin, Ll, bMin, Lb, i, j)

            # check if we have already counted the cluster
            counted = _counted_cluster(df_box, all_clusters, members)

            if not edge and not counted:
                df_members = df_box.loc[members].assign(
                    cluster_label=n_label).assign(dens=dens)
                all_clusters = all_clusters.append(df_members)
                n_label = n_label + 1

    # save a dataframe with all the found clusters
    name = "dr2_clusters_L{}_minPts{}.csv".format(L, min_pts)
    columns = header + ['cluster_label', 'dens']
    all_clusters[columns].to_csv(os.path.join(out_dir, name))

    with open(os.path.join(out_dir, 'postprocess_stats_L{}_minPts{}.csv'.format(L, min_pts)), "a") as myfile:
        myfile.write(str(_memory_usage()) + '\n')


def _compute_boxes(in_file, stats_file, L, min_pts):

    # Compute number of boxes and actual size
    nl = int(np.floor((lMaxim - lMinim) / L))
    nb = int(np.floor((bMaxim - bMinim) / L))
    Ll = (lMaxim - lMinim) / nl
    Lb = (bMaxim - bMinim) / nb

    shifts = [0., 1./3., 2./3.]
    db_results = []

    for reps in range(repetitions):
        for shift in shifts:
            db_results = _compute_shift(in_file, stats_file, shift, Lb, nb, Ll, nl,
                                        db_results, min_pts)
            if count >= MAX_BOXES:
                return db_results
    return db_results


def _compute_shift(in_file, stats_file, shift, Lb, nb, Ll, nl, db_results, min_pts):
    lMin = lMinim - shift * Ll
    bMin = bMinim - shift * Lb
    if shift == 0.:
        nll = nl
        nbb = nb
    else:
        # If the boxes are shifted, add one box to cover all the region
        nll = nl + 1
        nbb = nb + 1

    print("Computing ", nll * nbb, " regions. Time: ",
          "{0:.2f}".format(time.time() - s_time), flush=True)

    for i in range(nll):
        for j in range(nbb):
            box_results = _box_computation(in_file, stats_file, min_pts, Ll, Lb, lMin,
                                           bMin, i, j)
            db_results.append((shift, i, j, lMin, Ll, bMin, Lb) + tuple(box_results))
            global count
            count += 1
            if count >= MAX_BOXES:
                return db_results

    return db_results


@task(returns=5)
def _box_computation(in_file, stats_file, min_pts, Ll, Lb, lMin, bMin, i, j):
    t_box_0 = time.time()
    df = _read_dataframe(in_file)
    # Standarisation of features (5-D)
    df_box = df.query('@lMin + @i*@Ll < l < @lMin + (@i+1)*@Ll and '
                      '@bMin + @j*@Lb < b < @bMin + (@j+1)*@Lb')
    samples_size = len(df_box)
    data_scaled = StandardScaler().fit_transform(np.array(df_box[header]))
    df_scaled = pd.DataFrame(data=data_scaled,
                             columns=df_box.columns, dtype=float)
    dens = _get_density(df_box, samples_size)
    k_neighbors = min_pts - 1
    epsilon = _compute_epsilon(df_box, df_scaled, k_neighbors, samples_size)
    db = DBSCAN(eps=epsilon, min_samples=min_pts)
    labels = db.fit(np.array(df_scaled)).labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    _save_box_stats(stats_file, lMin, i, j, samples_size, epsilon, n_clusters, _memory_usage(), time.time() - t_box_0)
    return df_box, dens, epsilon, n_clusters, labels

def _save_box_stats(stats_file, *args):
    with open(stats_file, "a") as myfile:
        myfile.write(','.join([str(arg) for arg in args]) + '\n')

def _memory_usage():  # maximum resident set size in kilobytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def _counted_cluster(df_box, all_clusters, members):
    return (any(df_box.loc[members, 'l'].isin(all_clusters['l']))) or (
        any(df_box.loc[members, 'b'].isin(all_clusters['b'])))


def _in_edge(df_box, members, lMin, Ll, bMin, Lb, i, j):
    # check if any member falls in the edge.
    minl = min(df_box.loc[members, 'l'])
    maxl = max(df_box.loc[members, 'l'])
    minb = min(df_box.loc[members, 'b'])
    maxb = max(df_box.loc[members, 'b'])

    return minl < (lMin + i * Ll + tol) or maxl > (
            lMin + (i + 1) * Ll - tol) or minb < (
                   bMin + j * Lb + tol) or maxb > (
                   bMin + (j + 1) * Lb - tol)


def _get_members(df_box, c, labels):
    return df_box.iloc[[t for (t, lab) in enumerate(labels) if lab == c]].index


def _get_density(df_box, samples_size):
    l_max = df_box['l'].max()
    l_min = df_box['l'].min()
    b_max = df_box['b'].max()
    b_min = df_box['b'].min()

    return float(samples_size) / ((l_max - l_min) * (b_max - b_min))


# Computation of epsilon (parameter needed for DBSCAN)
def _compute_epsilon(df_box, df_scaled, k, n):
    """
    Compute epsilon for a given

    Input
        df_box: Non-Standarised DataFrame where to compute epsilon
        k: number of nearest neighbour to take into account in the computation

    Output
        espilon
    """
    # Compute kth nearest neighbour distance to each source
    neighs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(
        np.array(df_scaled))
    dist, ind = neighs.kneighbors(df_scaled)
    k_dist = [dist[t, k - 1] for t in range(len(dist))]
    # Here we want to draw a sample of only field stars (no clusters)
    min_k_dist_sim = []
    # Repeat the process n_simulations times to reduce random effects
    for s in range(n_simulations):
        m = []
        for par in header:
            # Capture the distribution of the field
            kernel = gaussian_kde(df_box[par])
            # Random resampling from the captured distribution
            # (as it is a random resampling, no clusters are expected)
            resamp = kernel.resample(n)[0]
            m.append(resamp)
        # Standarise the features the same way we did for the real data
        df_sim = pd.DataFrame(data=np.matrix(m).T, columns=df_box.columns,
                              dtype=float)
        data_sim_scaled = StandardScaler().fit_transform(
            np.array(df_sim[header]))
        df_sim_scaled = pd.DataFrame(data=data_sim_scaled,
                                     columns=df_sim.columns, dtype=float)
        neighs_sim = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(
            np.array(df_sim_scaled))  # scaled
        dist_sim, ind_sim = neighs_sim.kneighbors(df_sim_scaled)  # scaled
        k_dist_sim = [dist_sim[t, k - 1] for t in range(len(dist_sim))]
        min_k_dist_sim.append(np.min(k_dist_sim))
    e_max = np.mean(min_k_dist_sim)
    # Take the mean between the minimum of the (real data) kth nearest
    # neighbour distribution (where we expect low values corresponding to the
    # data) and the minimum of (simulated) field stars
    epsilon = (1. / 2.) * (np.min(k_dist) + e_max)
    return epsilon


def _read_dataframe(path):
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.pkl'):
        df = pd.read_pickle(path)
    elif path.endswith('.npy'):
        df = pd.DataFrame(np.load(path))
    else:
        raise ValueError
    df.columns = header
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", metavar="OUT_LOCATION", type=str, required=True)
    parser.add_argument("input_data", type=str)
    parser.add_argument("-L", metavar="BOX_SIZE", type=int)
    parser.add_argument("-m", metavar="MIN_PTS", type=int)
    args = parser.parse_args()

    if args.L is not None:
        global box_sizes
        box_sizes = [args.L]
    if args.m is not None:
        global min_samples
        min_samples = [args.m]

    _do_clustering(args.input_data, args.o)


if __name__ == "__main__":
    main()
