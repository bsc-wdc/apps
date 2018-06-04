from sklearn.metrics import r2_score

from pycompss.api.api import compss_wait_on
from pycompss.api.task import task
from pycompss.api.parameter import INOUT

from sklearn.ensemble import forest as skforest
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import DTYPE, DOUBLE    # using sklearn version 0.19.1
from sklearn.utils import check_random_state, check_array, compute_sample_weight
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import DataConversionWarning, NotFittedError

import warnings
from warnings import warn

import numpy as np
import scipy as sp
from scipy.sparse import issparse
from scipy.sparse import hstack as sparse_hstack

import sklearn
if sklearn.__version__ != '0.19.1':
    warn('The loaded version of sklearn is ' + sklearn.__version__ + ' but 0.19.1 may be necessary. '
         'This module accesses and modifies protected variables of sklearn_0.19.1, that may no longer '
         'exist in later versions.')

MAX_INT = np.iinfo(np.int32).max


def _generate_sample_indices(random_state, n_samples):
    """Private function used to _parallel_build_trees function."""
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)

    return sample_indices


def _generate_unsampled_indices(random_state, n_samples):
    """Private function used to forest._set_oob_score function."""
    sample_indices = _generate_sample_indices(random_state, n_samples)
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]

    return unsampled_indices


@task(returns=np.ndarray)
def _distributed_apply(tree, X):
    check_is_fitted(tree, 'tree_')
    X = tree._validate_X_predict(X, True)
    return tree.tree_.apply(X)


@task(tree=INOUT, returns=DecisionTreeRegressor)
def _distributed_build_trees(tree, bootstrap, X, y, sample_weight, tree_idx, n_trees,
                             verbose=0, class_weight=None):
    """Private function used to fit a single tree in a distributed way."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    if bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        indices = _generate_sample_indices(tree.random_state, n_samples)
        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight == 'subsample':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                curr_sample_weight *= compute_sample_weight('auto', y, indices)
        elif class_weight == 'balanced_subsample':
            curr_sample_weight *= compute_sample_weight('balanced', y, indices)

        tree.fit(X, y, sample_weight=curr_sample_weight, check_input=False)
    else:
        tree.fit(X, y, sample_weight=sample_weight, check_input=False)

    return tree


@task(returns=sp.sparse.csr.csr_matrix)
def _distributed_decision_path(tree, X):
    return tree.decision_path(X, check_input=False)


@task(returns=list)
def _distributed_predict(estimator, X):
    prediction = estimator.predict(X, check_input=False)
    return prediction


@task(returns=np.ndarray)
def _distributed_feature_importances_(tree):
    return tree.feature_importances_


@task(returns=(list, list))
def _distributed_set_oob_score(X, estimator, n_samples, n_outputs_):
    unsampled_indices = _generate_unsampled_indices(
        estimator.random_state, n_samples)
    p_estimator = estimator.predict(
        X[unsampled_indices, :], check_input=False)

    if n_outputs_ == 1:
        p_estimator = p_estimator[:, np.newaxis]

    return p_estimator, unsampled_indices


@task(returns=np.ndarray)
def _distributed_validate_X_predict(estimator, X):
    return estimator._validate_X_predict(X, check_input=True)


class RandomForestRegressor(skforest.RandomForestRegressor):
    def __init__(self,
                 n_estimators=10,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(RandomForestRegressor, self).__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)
        self.n_outputs_ = None      # Initialized by fit(self, X, y)
        self.n_features_ = None     # Initialized by fit(self, X, y)
        self.estimators_ = []

    def decision_path(self, X):
        """Return the decision path in the forest

        .. versionadded:: 0.18

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        indicator : sparse csr array, shape = [n_samples, n_nodes]
            Return a node indicator matrix where non zero elements
            indicates that the samples goes through the nodes.

        n_nodes_ptr : array of size (n_estimators + 1, )
            The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]]
            gives the indicator value for the i-th estimator.

        """
        X = self._validate_X_predict(X)
        indicators = []
        for tree in self.estimators_:
            indicators.append(_distributed_decision_path(tree, X))

        indicators[:] = [compss_wait_on(ind) for ind in indicators]

        n_nodes = [0]
        n_nodes.extend([i.shape[1] for i in indicators])
        n_nodes_ptr = np.array(n_nodes).cumsum()

        return sparse_hstack(indicators).tocsr(), n_nodes_ptr

    def fit(self, X, y, sample_weight=None):
        """Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Returns self.
        """
        # Validate or convert input data
        X = check_array(X, accept_sparse="csc", dtype=DTYPE)
        y = check_array(y, accept_sparse='csc', ensure_2d=False, dtype=None)
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        # Remap output
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Check parameters
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = []
            for i in range(n_more_estimators):
                tree = self._make_estimator(append=False,
                                            random_state=random_state)
                trees.append(tree)

            # Distributed loop.
            for i, t in enumerate(trees):
                _distributed_build_trees(
                    t, self.bootstrap, X, y, sample_weight, i, len(trees),
                    verbose=self.verbose, class_weight=self.class_weight)

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted values.
        """
        check_is_fitted(self, 'estimators_')

        X_np = self._validate_X_predict(X)

        # Sum the prediction for every tree.
        predicts_reduction = []
        for e in self.estimators_:
            predicts_reduction.append(_distributed_predict(e, X_np))

        while len(predicts_reduction) > 1:
            length = len(predicts_reduction)
            for i in range(length / 2):
                predicts_reduction[i] = _distributed_sum(predicts_reduction[i], predicts_reduction[-(i+1)])
            predicts_reduction = predicts_reduction[: length / 2 + length % 2]

        y_hat = compss_wait_on(predicts_reduction[0])

        y_hat /= len(self.estimators_)

        return y_hat

    def _validate_X_predict(self, X):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        return _distributed_validate_X_predict(self.estimators_[0], X)

    def apply(self, X):
        """Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array_like, shape = [n_samples, n_estimators]
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        """
        X = self._validate_X_predict(X)
        results = []
        for tree in self.estimators_:
            results.append(_distributed_apply(tree, X))
        results[:] = [compss_wait_on(re) for re in results]
        return np.array(results).T

    def _set_oob_score(self, X, y):
        """Compute out-of-bag scores"""
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_samples = y.shape[0]

        p_estimator_array = []
        unsampled_indices_array = []

        predictions = np.zeros((n_samples, self.n_outputs_))
        n_predictions = np.zeros((n_samples, self.n_outputs_))

        for estimator in self.estimators_:
            p_estimator, unsampled_indices = _distributed_set_oob_score(X, estimator, n_samples, self.n_outputs_)
            p_estimator_array.append(p_estimator)
            unsampled_indices_array.append(unsampled_indices)

        for i in range(len(p_estimator_array)):
            p_estimator_array[i] = compss_wait_on(p_estimator_array[i])
            unsampled_indices_array[i] = compss_wait_on(unsampled_indices_array[i])

        for i in range(len(p_estimator_array)):
            predictions[unsampled_indices_array[i], :] += p_estimator_array[i]
            n_predictions[unsampled_indices_array[i], :] += 1

        if (n_predictions == 0).any():
            warn("Some inputs do not have OOB scores. "
                 "This probably means too few trees were used "
                 "to compute any reliable oob estimates.")
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions
        self.oob_prediction_ = predictions

        if self.n_outputs_ == 1:
            self.oob_prediction_ = \
                self.oob_prediction_.reshape((n_samples, ))

        self.oob_score_ = 0.0

        for k in range(self.n_outputs_):
            self.oob_score_ += r2_score(y[:, k], predictions[:, k])

        self.oob_score_ /= self.n_outputs_

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        check_is_fitted(self, 'estimators_')

        all_importances = []

        for tree in self.estimators_:
            all_importances.append(_distributed_feature_importances_(tree))

        all_importances[:] = [compss_wait_on(imp) for imp in all_importances]

        return sum(all_importances) / len(self.estimators_)


@task(returns=list, priority=True)
def _distributed_sum(a, b):
    """Reduce"""
    return a + b