from __future__ import print_function

__author__ = 'p.glock'

from mpi4py import MPI
import numpy as np
import logging
from sklearn.svm import SVC, LinearSVC
from inspect import isclass
from functools import partial
import argparse
from sklearn.datasets import load_svmlight_file
from numbers import Integral

import sys
import time

world_comm = MPI.COMM_WORLD
base_classifier = SVC

debug = False

try:
    import pyextrae.mpi as pyextrae
except:
    pass

def dist_filter(X, y, k, **kwargs):
    from scipy.spatial.distance import pdist, squareform
    unique_y = list(set(y))
    neg_X = X[y == unique_y[0]]
    pos_X = X[y == unique_y[1]]

    neg_Dis = squareform(pdist(neg_X))
    pos_Dis = squareform(pdist(pos_X))

    neg_Dis = neg_Dis.mean(axis=1)
    pos_Dis = pos_Dis.mean(axis=1)

    neg_sort = neg_Dis.argsort()[:-1]
    pos_sort = pos_Dis.argsort()[:-1]

    res = np.vstack((neg_X[neg_sort[:k], :], pos_X[pos_sort[:k], :]))
    length = res.shape[0]
    labels = np.ones(length)
    n_length = neg_sort[:k].shape[0]
    labels[:n_length] = labels[:n_length] * -1
    return res, labels


def alpha_filter(X, y, k, alpha, **kwargs):

    unique_y = list(set(y))
    neg_X = X[y == unique_y[0]]
    pos_X = X[y == unique_y[1]]

    neg_alpha = alpha[y == unique_y[0]]
    pos_alpha = alpha[y == unique_y[1]]

    na_sort = np.abs(neg_alpha).argsort()[:-1]
    pa_sort = np.abs(pos_alpha).argsort()[:-1]

    res = np.vstack((neg_X[na_sort[:k], :],pos_X[pa_sort[:k], :]))
    length = res.shape[0]
    labels = np.ones(length)
    n_length = na_sort[:k].shape[0]
    labels[:n_length] = labels[:n_length] * -1
    return res, labels


def lagrangian_fast(SVs, sl, coef, kernel):
    set_sl = set(sl)
    assert len(set_sl) == 2, "Only binary problem can be handled"
    new_sl = sl.copy()
    new_sl[sl == 0] = -1

    C1, C2 = np.meshgrid(coef, coef)
    L1, L2 = np.meshgrid(new_sl, new_sl)
    double_sum = C1 * C2 * L1 * L2 * kernel(SVs)
    double_sum = double_sum.sum()
    W = -0.5 * double_sum + coef.sum()

    return W


def linear_kernel(x1):
    return np.dot(x1, x1.T)


def rbf_kernel(x, gamma=0.1):
    # Trick: || x - y || ausmultipliziert
    sigmaq = -1 / (2 * gamma)
    n = x.shape[0]
    K = np.dot(x, x.T) / sigmaq
    d = np.diag(K).reshape((n, 1))
    K = K - np.ones((n, 1)) * d.T / 2
    K = K - d * np.ones((1, n)) / 2
    K = np.exp(K)
    return K


def root_print(s, root=0, comm=world_comm, file=sys.stdout, should_print=True):
    if comm.rank == root:
        if should_print:
            print(s, file=file)


def setdiff(a, b):
    a_rows = a.view([('', a.dtype)] * a.shape[1])
    b_rows = b.view([('', b.dtype)] * b.shape[1])

    return np.setdiff1d(a_rows, b_rows).view(a.dtype).reshape(-1, a.shape[1])


def unique(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return a[ui]


def calc_levels(n):
    """ Calculate the number of levels for a binary cascade for a starting level with n processes/models.
    This is done by finding the highest i, so that 2^i <= n, i in R.

    Arguments:
        n (int) : positive integer, number of processes for the first level

    Return:
        int: i + 1
    """
    count = 0
    while 2 ** count <= n:
        count += 1
    if debug:
        print("Levels %s" % count)
    return count


def create_binary_cascade(n=world_comm.size, base=SVC, mode="regular", **kwargs):
    """ Create a binary cascade SVM with n processes at the first level.

    Arguments:
        n (int): number of models/processes of the first level of the cascade SVM.
        mode : string, mode of the cascade. At the moment only 'regular' is supported. Regular: n = 2^i, additional
            processes are ignored.

    Return:
        Cascade: a binary cascadeSVM
    """
    num_lvl = 0
    if mode == "regular":
        # n = max(2^i) <= n
        num_lvl = calc_levels(n)
        n = 2 ** (num_lvl - 1)

    procs = range(n)
    cascade = Cascade(base, **kwargs)

    for i in range(num_lvl):
        lvl = Level(i, procs)
        cascade.add_level(lvl)
        if len(procs) > 1:
            br = Bridge()
            next = procs[::2]
            for index in range(0, len(procs), 2):
                element = BridgeElement([procs[index], procs[index + 1]], procs[index])
                br.add(element)
            cascade.add_bridge(br)
            procs = next

    return cascade


class Cascade(object):
    """ A CascadeSVM contains n levels and n-1 bridges.
    A level are all models/processes that run at the same time. A bridge is used to pass the result (support vectors)
    to the next level.
    Currently SVC is used for a single SVM model.

    Arguments:
        base (class): SVM classifier used as a base for the cascadeSVM.
        **kwargs : Arguments used for the SVM classifier.

    Attributes:
        levels (list): list of levels
        bridges (list): list of bridges
        base (class): base SVM classifier
    """

    def __init__(self, base=SVC, timeit=True, **kwargs):
        self.levels = []
        self.bridges = []
        self.initialized = False
        self.base = None
        self.last_supp = None
        self.last_label = None
        self.lastW = None
        self.times = {"mpi": 0., "computation": 0.}
        self.timeit = timeit
        self.feedback_filter = None

        if isclass(base):
            self.base = partial(base, **kwargs)
        else:
            raise Exception("A Class is needed!")

    def init_data(self, filename, file_format="libsvm", shuffle=False):
        """ Initialize X and y. They are scattered among the processes of the first level.

        Arguments:
            filename (str): File containing training data.
            file_format (str): Format of the file. Either 'libsvm'(default) or 'hdf5'(parallel I/O).

        Return:
            None
        """

        if self.timeit:
            t0 = time.time()

        # if world_comm.rank == 0:
        #
        # comb = np.concatenate((X, y.reshape((-1, 1))), axis=1)
        # comb = unique(comb)
        #     np.random.shuffle(comb)
        #     X = comb[:, :-1].copy()
        #     y = comb[:, -1].copy()
        #     y = y.astype(int)

        if file_format == "libsvm":
            if world_comm.rank == 0:
                X, y = load_svmlight_file(filename)
                X = X.toarray()
                y = y.astype('int')
                if shuffle:
                    from data import shuffle_data

                    X, y = shuffle_data(X, y)
            else:
                X = None
                y = None

            self.levels[0].scatter_data(X, y)
            if self.timeit:
                t1 = time.time() - t0
                #root_print("Initializing data by scattering: {s}s".format(s=t1))
                self.times["mpi"] += t1
        elif file_format == "hdf5":
            from data import h5_pread

            X, y = h5_pread(filename, self.levels[0].level_comm)
            self.levels[0].X = X
            self.levels[0].baseX = X.copy()
            self.levels[0].y = y
            self.levels[0].basey = y.copy()

        elif file_format == "csv":
            if world_comm.rank == 0:
                data = np.loadtxt(filename, delimiter=',')
                X = np.ascontiguousarray(data[:, :-2], copy=True)
                y = data[:, -1].astype('int')
                if shuffle:
                    from data import shuffle_data

                    X, y = shuffle_data(X, y)
            else:
                X = None
                y = None
            self.levels[0].scatter_data(X, y)
            if self.timeit:
                t1 = time.time() - t0
                #root_print("Initializing data by scattering: {s}s".format(s=t1))
                self.times["mpi"] += t1

        self.initialized = True


    def _check_converged(self, comm):
        if comm != MPI.COMM_NULL:
            conv = np.array(0)
            if self.lastW is None:
                conv = np.array(0)
                root_print("Conv1. values %s" % conv)
            else:
                first_level = self.levels[0]
                if first_level.model.kernel == "linear":
                    kernel = linear_kernel
                elif first_level.model.kernel == "rbf":
                    kernel = partial(rbf_kernel, gamma=first_level.model.gamma)

                W = lagrangian_fast(first_level.support_vectors, first_level.support_labels,
                                    first_level.model.dual_coef_, kernel)
                max_W = np.array(0.)

                first_level.level_comm.Reduce(W, max_W, MPI.MAX, root=first_level.root)

                if world_comm.rank == first_level.global_root:
                    if np.abs((max_W - self.lastW) / self.lastW) < 10 ** -3:
                        root_print((max_W - self.lastW) / self.lastW)
                        root_print("Conv. values %s" % (root_print((max_W - self.lastW) / self.lastW)))
                        conv = np.array(1)
                    else:
                        root_print(np.abs((max_W - self.lastW) / self.lastW))

                first_level.level_comm.Bcast(conv, root=first_level.root)

            if conv == 1:
                return True
            else:
                return False

        else:
            return True


    def cascade_iteration(self, convergence_check=False):
        """ Performs one iteration of a cascadeSVM fit.
        Each level is trained and the resulting support vectors are transferred to the next level as input data.

        Return:
            None
        """
        index = 0
        while index < len(self.levels):
            lvl = self.levels[index]
            try:
                # Timing
                if self.timeit:
                    t0 = time.time()

                # train a level
                lvl.fit(self.base)

                # Timing
                if self.timeit:
                    t1 = time.time() - t0
                    root_print("Fit level [{i}]: {s} s".format(i=index, s=t1))
                    self.times["computation"] += t1

                # do feedback if intended. If feedback is done 'feedback' is True else False
                feedback = lvl.feedback(self.levels[0], self.feedback_filter)



                # if crossfeedback has been done, start at level 0 again
                if feedback:
                    index = 0
                    continue

                if index == 0 and convergence_check:
                    # convergence check
                    if self._check_converged(self.levels[0].level_comm):
                        return True

                # send data to next level
                if index < len(self.bridges):
                    if self.timeit:
                        t0 = time.time()

                    self.bridges[index].to_next(lvl, self.levels[index + 1])

                    if self.timeit:
                        t1 = time.time() - t0
                        root_print("Sending data to next level[{i}]: {s} s".format(i=index + 1, s=t1))
                        self.times["mpi"] += t1

                # if no feedback has been done increase level by one
                if not feedback:
                    index += 1

            except Exception as e:
                print("[{r}] Level {i}".format(r=world_comm.rank, i=index))
                import traceback

                traceback.print_exc()
                world_comm.Abort()

        return False

    def cascade_fit(self, repeat=1):

        if isinstance(repeat, Integral):
            for i in range(repeat):
                root_print("***** Iteration {n} *****".format(n=i + 1))
                self.cascade_iteration(convergence_check=False)
                if i < repeat-1:
                    self.broadcast_result()
        elif repeat == "converge":
            count = 1
            while True:
                root_print("***** Iteration {n} *****".format(n=count))
                count += 1
                converged = self.cascade_iteration(convergence_check=True)
                if converged:
                    break
                self.broadcast_result()

    def predict(self, X, root=0):
        if world_comm.rank in self.levels[0].procs:
            model = self.levels[-1].model
            communicator = self.levels[0].level_comm
            return mpi_predict(model, X, comm=communicator, root=root)

    def score(self, X, y, root=0):
        if world_comm.rank in self.levels[0].procs:
            model = self.levels[-1].model
            communicator = self.levels[0].level_comm
            acc = mpi_score(model, X, y, comm=communicator, root=root)

            return acc


    def broadcast_result(self):
        if world_comm.rank in self.levels[0].procs:
            last_level = self.levels[-1]

            if self.timeit:
                t0 = time.time()

            self.levels[0].X, self.levels[0].y, self.last_supp, self.last_label = _combine_data(self.levels[0].baseX,
                                                                                                last_level.support_vectors,
                                                                                                self.levels[0].basey,
                                                                                                last_level.support_labels,
                                                                                                True, self.levels[
                                                                                                    0].level_comm,
                                                                                                self.levels[0].root)

            if world_comm.rank in last_level.procs:

                if last_level.model.kernel == "linear":
                    kernel = linear_kernel
                elif last_level.model.kernel == "rbf":
                    if debug:
                        print(last_level.model.gamma)
                    kernel = partial(rbf_kernel, gamma=last_level.model.gamma)

                self.lastW = lagrangian_fast(last_level.support_vectors, last_level.support_labels,
                                             last_level.model.dual_coef_,
                                             kernel)
                self.lastW = np.array(self.lastW)
            else:
                self.lastW = np.array(0.)

            self.levels[0].level_comm.Bcast(self.lastW, root=self.levels[0].root)

            if self.timeit:
                t1 = time.time() - t0
                root_print("Broadcasting results to first level: {s} s".format(s=t1))
                self.times["computation"] += t1

        else:
            return None


    def save_model(self, f, root=0):
        """ Save the computed SVM.

        Arguments:
            f (str): filename as a string, to save the model.
            root (int): rank that saves the file.

        Return:
            None
        """

        if world_comm.rank == root:
            import pickle

            with open(f, "w") as ff:
                pickle.dump(self.levels[-1].model, ff)

    def add_level(self, level):
        """ Add a new level to the cascade.

        Arguments:
            level (Level): Level, Level to be added.

        Return:
            bool: True if adding was successful, False if not.
        """
        if type(level) is Level:
            self.levels.append(level)
            return True
        else:
            return False

    def add_bridge(self, b):
        """ Add a new bridge to the cascade.

        Arguments:
            b (Bridge): Bridge, Bridge to be added.

        Return:
            bool: True if adding was successful, False if not.
        """
        if type(b) is Bridge:
            self.bridges.append(b)
            return True
        else:
            return False

    @property
    def size(self):
        """ Returns the size of the Cascade. (Number of Levels)

        Return:
            int: size of the Cascade/number of levels.
        """
        return len(self.levels)

    def __str__(self):
        printable = "Cascade SVM, number of levels: {l}, number of bridges: {b}\nLevels:\n".format(l=len(self.levels),
                                                                                                   b=len(self.bridges))
        for l in self.levels:
            printable += str(l) + "\n"
        return printable


class Level(object):
    """A level contains all processes that are trained at the same time.

    The level object also handles the data for the different processes.
    E.g. the X attribute, which handles the training samples. If the process is not in the level, it is None.
    X may be different for every process in the level.
    The processes are set and a communicator is created, which is needed for scattering the training data.
    The level object can be created on all processes but only the ones listed in self.procs are functional.
    The rest only returns None if a method is called.

    Attributes:
        X (array): numpy.array, sample data for each process of the level.
        y (array): numpy.array, sample class labels for each process.
        procs (list): list of processors in the level, global ranks.
        level_rank (int): level rank of the process, -1 if not in the level.
        global_root (int): global rank of the root process. Default: 0.
        support_vectors (array): numpy.array, init with None.
            After training set to the support vectors of the model.
        support_labels (array): numpy.array, init with None. After training set to the support labels of the model.
        model : Classifier. The default classifier is a SVC.
        n_feedback (int): Number of feedbacks done at this level.

    """

    def __init__(self, rank, processes):
        self.rank = rank
        self.procs = processes
        group = world_comm.Get_group()
        new_group = group.Incl(self.procs)
        self.level_comm = world_comm.Create(new_group)

        if self.level_comm != MPI.COMM_NULL:
            self.level_rank = self.level_comm.Get_rank()
        else:
            self.level_rank = -1

        self.global_root = 0
        global_to_level = np.zeros(world_comm.size, dtype=np.int64)
        world_comm.Allgather(np.array(self.level_rank, dtype=np.int64), global_to_level)
        self.global_to_level = global_to_level
        self.X = None
        self.y = None
        self.baseX = None
        self.basey = None
        self.support_vectors = None
        self.support_labels = None
        self.model = None
        self.n_feedback = 0
        self.max_feedback = 0

    def feedback(self, target_level, feedback_filter=None):

        if self.n_feedback < self.max_feedback:
            root_print("##### Feedback from level {} to level {} #####".format(self.rank, target_level.rank))
            self.n_feedback += 1
            for root in self.procs:
                exclude_procs = []#range(root+1, root+2**self.rank)

                if target_level.level_comm.size < world_comm.size:
                    exclude_procs.extend(range(target_level.level_comm.size, world_comm.size))

                group = world_comm.Get_group()
                new_group = group.Excl(exclude_procs)
                comm = world_comm.Create(new_group)

                # rank of the root in the new communicator
                grp_root = MPI.Group.Translate_ranks(group, [root], new_group)[0]

                #print("world root: {r}, group root: {g}".format(r=root,g=grp_root))

                if world_comm.rank == root:
                    if feedback_filter is None:
                        feed_X = self.support_vectors
                        feed_y = self.support_labels
                    else:
                        feed_X, feed_y = feedback_filter(self.support_vectors, self.support_labels, alpha=self.model.dual_coef_.flatten())
                else:
                    feed_X = None
                    feed_y = None

                if comm != MPI.COMM_NULL:
                    #print("RANK {} GETS FEEDBACK".format(world_comm.rank))
                    target_level.X, target_level.y = _combine_data(target_level.baseX, feed_X, target_level.basey, feed_y, comm=comm, root=grp_root, ret_last=False)

            return True
        else:
            return False

    @property
    def root(self):
        """ Transforms the global root to the level root.

        Return:
            int: level root rank.
        """
        return self.global_to_level[self.global_root]

    def _to_level(self, gl):
        """ Transforms a global rank to a level rank.

        Return:
            int: level rank of the global rank.
        """
        return self.global_to_level[gl]

    def scatter_data(self, X, y):
        """ Scatter X and y to all processes of the level.

        If called by a process, which is not in the level, nothing is done and None is returned.

        Arguments:
            X (array): numpy.array, 2 dimensional, sample data which is scattered.
            y (array): numpy.array, sample class labels which are scattered.

        Return:
            None
        """
        if world_comm.rank in self.procs:
            sX, sy = _scatter_data(X, y, comm=self.level_comm, root=self.root)
            self.X = sX
            self.y = sy
            self.baseX = sX.copy()
            self.basey = sy.copy()
        else:
            return None

    def fit(self, base):
        """ Trains the models on the given data.

        If X or y are None an Exception is raised. If the process is not in the level nothing is done.
        The support vectors and labels as well as the model are saved on the attributes.
        """
        if world_comm.rank in self.procs:
            if self.X is None or self.y is None:
                raise Exception("Data not available")

            self.support_vectors, self.support_labels, self.model = _mpi_train(base, self.X, self.y, ret="both")
            # print("[{r}]:\n{v}".format(r=world_comm.rank,v=self.support_vectors))
            # print("[{r}]:\n{v}".format(r=world_comm.rank,v=self.X))
        else:
            return None

    def score(self, X, y):
        """ Predict X and calculate the accuracy.

        Arguments:
            X (array): numpy.array, 2 dimensional, samples to predict.
            y (array): numpy.array, sample labels.
        """
        if world_comm.rank in self.procs:
            return self.model.score(X, y)
        else:
            return None

    def __str__(self):
        return str(self.procs)


class BridgeElement(object):
    """ A BridgeElement handles one communicator of a bridge.

    For a binary Cascade it has two source processors and one destination process.
    The destination may also be a source.

    Arguments:
        src (list): list of global processor ranks.
        root (int): global rank of root/destination.

    Attributes:
        procs (list): list of processes in the bridgeElement, global ranks.
        global_root (int): global rank of the root process.
        bridge_comm (Communicator): MPI communicator for the bridgeElement. NULL_COMM if process is not part of the bridge.
        bridge_rank (int): bridge rank of the process, -1 if process is no part.

    """

    def __init__(self, src, root):

        src = list(src)
        if root in src:
            self._root_incl = True
            self.procs = src
        else:
            src.append(root)
            self.procs = src
            self._root_incl = False

        self.global_root = root

        group = world_comm.Get_group()
        new_group = group.Incl(self.procs)
        self.bridge_comm = world_comm.Create(new_group)
        if self.bridge_comm != MPI.COMM_NULL:
            self.bridge_rank = self.bridge_comm.Get_rank()
        else:
            self.bridge_rank = -1

        global_to_bridge = np.zeros(world_comm.size, dtype=np.int64)
        world_comm.Allgather(np.array(self.bridge_rank, dtype=np.int64), global_to_bridge)
        self.global_to_bridge = global_to_bridge


    @property
    def root(self):
        """ Transforms the global root to the bridge root.

        Return:
            int: bridge root rank.
        """
        return self.global_to_bridge[self.global_root]

    def to_next(self, orig_X, orig_y):
        """ Sends the data from the source processes to the destination process and combines them to one array.

        Does nothing if rank is not part of the bridgeElement.

        Arguments:
            orig_X (array): numpy.array, vectors to be send to the destination.
            orig_y (array): numpy.array, labels to be send to the destination.

        """
        if world_comm.rank in self.procs:
            new_X, new_y = _all_to_one(support_vectors=orig_X, support_labels=orig_y, comm=self.bridge_comm,
                                       root=self.root)
            return new_X, new_y
        else:
            return None

    def __str__(self):
        return "{src} -> {dest}\n".format(src=self.procs, dest=self.global_root)


class Bridge(object):
    """ A Bridge contains a list of BridgeElements and connects to levels.

    """

    def __init__(self):
        self.parts = []

    def add(self, bridge_element):
        """ Add a bridgeElement to the bridge

        Arguments:
            bridge_element (BridgeElement): Element to be added.

        Return:
            bool: True if successful, False if not.
        """
        if type(bridge_element) is BridgeElement:
            self.parts.append(bridge_element)
            return True
        else:
            return False

    def to_next(self, prev_level, next_level):
        """ Sends prev_level.X and prev_level.y to next_level for the given processes.

        Arguments:
            prev_level (Level): previous level
            next_level (Level): next level
        """
        for p in self.parts:
            res = p.to_next(prev_level.support_vectors, prev_level.support_labels)
            if res is not None:
                next_level.X, next_level.y = res

    def __str__(self):
        s = "Bridge:\n"
        for b in self.parts:
            s += str(b)
        return s


def _calc_slices(X, comm=world_comm):
    """Calculate the slices of data for each process.

    Arguments:
        X : numpy.array, 2 dimensional

    Return:
        number of rows each process gets as a numpy array
    """

    n_rows = X.shape[0]
    slices = [n_rows // comm.size for i in range(comm.size)]
    count = n_rows % comm.size
    for i in range(count):
        slices[i] += 1

    return np.array(slices, dtype=np.int64)


def _scatter_samples(X, comm=world_comm, root=0):
    # cast X to numpy array if it is not None
    if X is not None and type(X) != np.ndarray:
        X = np.array(X)

    if comm.rank == root:
        slices = _calc_slices(X, comm=comm)
        n_features = np.array(X.shape[1], dtype=int)
    else:
        slices = np.zeros(comm.size, dtype=np.int64)
        n_features = np.zeros(1, dtype=int)

    # Broadcast information for scatterv
    comm.Bcast(slices, root=root)
    comm.Bcast(n_features, root=root)

    # slices and pos for samples (2d has to be considered)
    data_slices = slices * n_features
    data_pos = np.array([sum(data_slices[:i]) for i in range(comm.size)], dtype=int)

    # number of rows for each process
    row_cnt = slices[comm.rank]

    # allocate memory for splitted samples and scatter it
    split_X = np.zeros((row_cnt, n_features.item()), dtype=float)
    comm.Scatterv([X, data_slices, data_pos, MPI.DOUBLE], split_X, root=root)

    return split_X


def _scatter_labels(y, comm=world_comm, root=0):
    if y is not None and type(y) != np.ndarray:
        y = np.array(y)

    if comm.rank == root:
        slices = _calc_slices(y, comm=comm)

    else:
        slices = np.zeros(comm.size, dtype=np.int64)

    # Broadcast information for scatterv
    comm.Bcast(slices, root=root)

    # pos for labels
    pos = np.array([sum(slices[:i]) for i in range(comm.size)], dtype=int)

    # number of rows for each process
    row_cnt = slices[comm.rank]

    # allocate memory for splitted samples and scatter it
    split_y = np.zeros(row_cnt, dtype=np.int64)
    comm.Scatterv([y, tuple(slices), tuple(pos), MPI.INT_INT], split_y, root=root)

    return split_y


def _scatter_data(X, y, comm=world_comm, root=0):
    """ MPI method, scatters data X and labels y to all processes.

    Arguments:
        X : array_like, data, each row is a sample and each column a feature
        y : array_like, y[i] is the class label of X[i]

    Return:
        splitted X and y for each process as numpy arrays.

    """

    split_X = _scatter_samples(X, comm, root)
    split_y = _scatter_labels(y, comm, root)
    return split_X, split_y


def _mpi_train(base, X, y, ret="data", **kwargs):
    """Train a model on all processes and return the support vectors.

    Arguments:
        base : classifier class, used to build n_procs models
        X : numpy.array, training samples, 2dimensional
        y : numpy.array, labels of the training samples
        kwargs : parameters for the classifier

    Return:
        tuple with support vectors and their labels
    """
    # create classifier and train it with given parameters
    clf = base(**kwargs)
    clf.fit(X, y)

    # get the support vectors
    support_vectors = X[clf.support_, :]
    support_labels = y[clf.support_].astype(np.int64)

    if ret == "model":
        return clf
    elif ret == "data":
        return support_vectors, support_labels
    elif ret == "both":
        return support_vectors, support_labels, clf


def _final_model(base, X, y, **kwargs):
    """Train the final model on process 0.

    Arguments:
        base : classifier class, used to build final models
        X : numpy.array, training samples, 2dimensional
        y : numpy.array, labels of the training samples
        kwargs : parameters for the classifier

    Return:
        on process 0 return final model, all other return None
    """

    if world_comm.rank == 0:
        clf = base(**kwargs)
        clf.fit(X, y)
    else:
        clf = None

    return clf


def _all_to_one_2d(vectors, comm=world_comm, root=0):
    n_features = vectors.shape[1]

    # gather number of support vectors for each process in an array.
    n_supp = np.zeros(comm.size, dtype=np.int64)
    comm.Allgather(np.array(vectors.shape[0]), n_supp)

    # allocate memory for all gathered support vectors at the root process.
    if comm.rank == root:
        all_vectors = np.zeros((n_supp.sum(), n_features))
    else:
        all_vectors = None

    # length of support vector array for every process (2d to 1d)
    data_slices = n_supp * n_features
    # starting position for support vectors, like labels
    data_pos = np.array([sum(data_slices[:i]) for i in range(comm.size)])
    # Gather support vectors at root process
    comm.Gatherv(vectors, [all_vectors, data_slices, data_pos, MPI.DOUBLE], root=root)

    return all_vectors


def _all_to_one_1d(labels, comm=world_comm, root=0):
    # gather number of support vectors for each process in an array.
    n_supp = np.zeros(comm.size, dtype=np.int64)
    comm.Allgather(np.array(len(labels)), n_supp)

    # allocate memory for all gathered support vectors at the root process.
    if comm.rank == root:
        all_labels = np.zeros(n_supp.sum(), dtype=np.int64)
    else:
        all_labels = None

    # starting position for the support labels of every process. (sum of labels from lower ranks)
    pos = np.array([sum(n_supp[:i]) for i in range(comm.size)])
    # gather support labels at root process
    comm.Gatherv(labels, [all_labels, n_supp, pos, MPI.INT_INT], root=root)

    return all_labels


def _all_to_one(support_vectors, support_labels, comm=world_comm, root=0):
    """Gathers all support vectors and labels at process root.

    Arguments:
        support_vectors : numpy.array, 2dimensional, support vectors of all processes
        support_labels : numpy.array, labels of support vectors
        comm : MPI communicator, communicator used.
        root (int): rank that is used as root.

    Return:
        gathered support vectors and labels on process 0, all other return None.
    """
    all_support_vectors = _all_to_one_2d(support_vectors, comm, root)
    all_support_labels = _all_to_one_1d(support_labels, comm, root)

    if comm.rank == root:
        comb = np.concatenate((all_support_vectors, all_support_labels.reshape((-1, 1))), axis=1)
        comb = unique(comb)
        all_support_vectors = comb[:, :-1].copy()
        all_support_labels = comb[:, -1].copy()
        all_support_labels = all_support_labels.astype(support_labels.dtype)

    return all_support_vectors, all_support_labels


def _combine_data(old_X, new_X, old_y, new_y, ret_last=False, comm=world_comm, root=0):
    """Send new_X and new_Y to all threads and combine it with the old ones.

    Arguments:
        old_X : numpy.array, 2 dimensional, old samples
        new_X : numpy.array, 2 dimensional, new samples
        old_y : numpy.array, 1 dimensional, old labels
        new_y : numpy.array, 1 dimensional, new labels

    Return:
        combined data of new and old samples and labels
    """

    n_features = old_X.shape[1]

    if comm.rank == root:
        n_rows = np.array(len(new_y), dtype=np.int64)
    else:
        n_rows = np.zeros(1, dtype=np.int64)

    comm.Bcast(n_rows, root=root)
    if comm.rank != root:
        new_X = np.zeros((n_rows.item(), n_features), dtype=np.float)
        new_y = np.zeros(n_rows, dtype=np.int64)

    comm.Bcast(new_X, root=root)
    comm.Bcast(new_y, root=root)

    comb_old = np.concatenate((old_X, old_y.reshape((-1, 1))), axis=1)
    comb_new = np.concatenate((new_X, new_y.reshape((-1, 1))), axis=1)

    diff = setdiff(comb_new, comb_old)
    y = diff[:, -1].copy()
    y = y.astype(new_y.dtype)
    X = diff[:, :-1]

    total_X = np.concatenate((old_X, X), axis=0)
    total_y = np.concatenate((old_y, y), axis=0)

    if debug:
        print("[%s] Old, new, and total sizes: %s, %s, %s", (comm.rank, old_X.size, new_X.size, total_X.size))
    if ret_last:
        return total_X, total_y, new_X, new_y
    else:
        return total_X, total_y


def mpi_predict(classifier, X, comm=world_comm, root=0):
    splitX = _scatter_samples(X, comm, root)
    classifier = comm.bcast(classifier, root=root)
    predicted = classifier.predict(splitX)

    all_predicted = _all_to_one_1d(predicted, comm, root)

    return all_predicted


def mpi_score(classifier, X, y, comm=world_comm, root=0):
    split_X, split_y = _scatter_data(X, y, comm, root)
    classifier = comm.bcast(classifier, root=root)

    score = classifier.score(split_X, split_y)

    # reduce accuracy with sum
    total_score = np.array(0.)
    comm.Reduce(score, total_score, MPI.SUM)
    if debug:
        print(total_score)
    total_score = total_score / comm.size

    return total_score


def mpi_fit(X, y, n_loops=5, comm=world_comm, logging_level=logging.DEBUG):
    """ Trains a cascade svm on the training data X with the labels y.

    Arguments:
        X : data
        y : labels
        comm : communicator
        logging_level : log output

    """

    split_X, split_y = _scatter_data(X, y, comm)
    train_X = split_X.copy()
    train_y = split_y.copy()

    for i in range(n_loops):
        sv, sl = _mpi_train(base_classifier, train_X, train_y)
        all_sv, all_sl = _all_to_one(sv, sl)

        final = _final_model(base_classifier, all_sv, all_sl)
        if comm.rank == 0:
            new_X = X[final.support_, :]
            new_y = y[final.support_]
        else:
            new_X = None
            new_y = None

        train_X, train_y = _combine_data(split_X, new_X, split_y, new_y, comm)

    return final


def testing():
    from sklearn.datasets import load_iris
    from sklearn.cross_validation import train_test_split
    #
    data = load_iris()
    data_train, data_test, target_train, target_test = train_test_split(data.data, data.target, test_size=0.33,
                                                                        random_state=7)  # random_state=7
    # clf = mpi_fit(data_train, target_train, n_loops=1)
    #
    # if world_comm.rank == 0:
    # acc = clf.score(data_test, target_test)
    # print("Accuracy: %f") % acc
    # print("number of support vectors: %d") % len(clf.support_)
    #
    # top = Level([0, 1])
    # bottom = Level([0])
    # bridgeE = BridgeElement([0, 1], 0)
    # bridge = Bridge()
    # bridge.add(bridgeE)
    #
    # if world_comm.rank in [0, 1]:
    # top.scatter_data(data_train, target_train)
    # top.fit()
    # # bottom.X,bottom.y = bridgeE.to_next(top.support_vectors,top.support_labels)
    # bridge.to_next(top, bottom)
    # bottom.fit()
    #
    # if world_comm.rank == 0:
    # acc = bottom.score(data_test,target_test)
    # print("Accuracy: %f") % acc
    # print("number of support vectors: %d") % len(bottom.support_vectors)

    csvm = create_binary_cascade(C=10, base=SVC)
    csvm.init_data(data_train, target_train)

    if world_comm.rank == 0:
        print(csvm)
    csvm.cascade_fit()

    if world_comm.rank == 0:
        acc = csvm.levels[-1].score(data_test, target_test)
        print("Accuracy: {acc}".format(acc=acc))
        print("number of support vectors: {vec}".format(vec=len(csvm.levels[-1].support_vectors)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    # testing method
    parser_test = subparsers.add_parser("test", help="A simple test case.")

    # training
    parser_train = subparsers.add_parser("train", help="train a cascade svm on training data.")
    parser_train.add_argument("-C", type=float, default=1., help="C )value of the base SVM.")
    parser_train.add_argument("-g", "--gamma", type=float, default=0.1, help="gamma value of the base SVM.")
    parser_train.add_argument("-k", "--kernel", type=str, default="rbf", help="kernel used for the SVM classifier.")
    parser_train.add_argument("data", type=str, help="training samples.")
    parser_train.add_argument("-s", "--save", type=str, help="save file for the calculated classifier.")
    parser_train.add_argument("--shuffle", action="store_true", help="shuffle training samples.")
    parser_train.add_argument("--score", type=str, help="test samples to calculate an accuracy.")
    parser_train.add_argument("-r", "--repeat", default=1, type=str,
                              help="number of iterations. 'converge' iterates until the convergence condition is True")
    parser_train.add_argument("-f", "--file_format", type=str, default="libsvm",
                              help="File format of the input file. 'libsvm'(default) or 'hdf5'(parallel I/O) or csv "
                                   "(label should be first element).")
    # which file format is used?
    file_format = parser_train.add_mutually_exclusive_group()
    file_format.add_argument("--libsvm", action="store_true", help="data is saved in libsvm format")
    file_format.add_argument("--hdf5", action="store_true", help="data is saves in hdf5 format")
    file_format.add_argument("--csv", action="store_true", help="data is saves in hdf5 format")

    # score
    parser_score = subparsers.add_parser("score", help="calculate score for a model on test data.")
    parser_score.add_argument("model", help="trained model.")
    parser_score.add_argument("data", help="test data.")

    args = parser.parse_args()

    if args.command == "test":
        testing()
    elif args.command == "train":
        t0 = time.time()

        if args.repeat == "converge":
            repeat = args.repeat
        else:
            try:
                repeat = int(args.repeat)
            except ValueError:
                print("Error: repeat must be either 'converge' or a positiv integer!")
                world_comm.Abort()

        cascade = create_binary_cascade(gamma=args.gamma, C=args.C)
        cascade.init_data(args.data, args.file_format, args.shuffle)
        cascade.levels[0].max_feedback = 1
        cascade.cascade_fit(repeat=repeat)
        
        t1 = time.time() - t0
        root_print("Elapsed Time: {s}s".format(s=t1), should_print=True)

        if args.score is not None and world_comm.rank == 0:
            X_test, y_test = load_svmlight_file(args.score)
            X_test = X_test.toarray()

            acc = cascade.levels[-1].model.score(X_test, y_test)
            print(acc)

        if args.save is not None:
            cascade.save_model(args.save)

    elif args.command == "score":
        if world_comm.rank == 0:
            import pickle

            X, y = load_svmlight_file(args.data)
            X = X.toarray()
            with open(args.model, "r") as f:
                clf = pickle.load(f)

            print(clf.score(X, y))
