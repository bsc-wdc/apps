"""Random forest for distributed computing using the COMPSs framework

This module adapts the RandomForestClassifier and RandomForestRegressor
classes from sklearn. Each tree is built and accessed in a parallel and
distributed way.

The module structure is the following:

- The ``BaseForest`` base class implements a common ``fit`` method for all
  the estimators in the module. The ``fit`` method of the base ``Forest``
  class calls the ``fit`` method of each sub-estimator on random samples
  (with replacement, a.k.a. bootstrap) of the training set.

  The init of the sub-estimator is further delegated to the
  ``BaseEnsemble`` constructor.

- The ``RandomForestClassifier`` and ``RandomForestRegressor`` base classes further
  implement the prediction logic by computing an average of the predicted
  outcomes of the sub-estimators.

Single and multi-output problems are both handled.

"""
# This module is a derivative work of the sklearn.ensemble.forest module. Original note:
# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joly Arnaud <arnaud.v.joly@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#
# License: BSD 3 clause

from __future__ import division

import warnings
from warnings import warn

from abc import ABCMeta, abstractmethod
import numpy as np
import scipy as sp
from pycompss.api.constraint import constraint
from scipy.sparse import issparse
from scipy.sparse import hstack as sparse_hstack


from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.externals import six
from sklearn.metrics import r2_score
from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor)
from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.utils import check_random_state, check_array, compute_sample_weight
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.ensemble.base import BaseEnsemble
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from pycompss.api.api import compss_wait_on
from pycompss.api.task import task
from pycompss.api.parameter import INOUT

import sklearn
if sklearn.__version__ != '0.19.1':
    warn('The loaded version of sklearn is ' + sklearn.__version__ + ' but 0.19.1 may be necessary. ')

__all__ = ["RandomForestClassifier",
           "RandomForestRegressor"]

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


@task(tree=INOUT, returns=DecisionTreeClassifier)
def _distributed_build_classifier_trees(tree, *args, **kwargs):
    return _distributed_build_trees(tree, *args, **kwargs)


@task(tree=INOUT, returns=DecisionTreeRegressor)
def _distributed_build_regression_trees(tree, *args, **kwargs):
    return _distributed_build_trees(tree, *args, **kwargs)


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


@task(returns=np.ndarray)
def _distributed_feature_importances_(tree):
    return tree.feature_importances_


@task(returns=list)
def _distributed_predict(estimator, X):
    prediction = estimator.predict(X, check_input=False)
    return prediction


@task(returns=list)
def _distributed_predict_proba(estimator, X):
    prediction = estimator.predict_proba(X, check_input=False)
    return prediction


@task(returns=(list, list))
def _distributed_set_oob_score_classifier(X, estimator, n_samples, n_outputs_):
    unsampled_indices = _generate_unsampled_indices(estimator.random_state, n_samples)
    p_estimator = estimator.predict_proba(X[unsampled_indices, :], check_input=False)

    if n_outputs_ == 1:
        p_estimator = [p_estimator]

    return p_estimator, unsampled_indices


@task(returns=(list, list))
def _distributed_set_oob_score_regressor(X, estimator, n_samples, n_outputs_):
    unsampled_indices = _generate_unsampled_indices(estimator.random_state, n_samples)
    p_estimator = estimator.predict(X[unsampled_indices, :], check_input=False)

    if n_outputs_ == 1:
        p_estimator = p_estimator[:, np.newaxis]

    return p_estimator, unsampled_indices


@task(returns=list)
def _distributed_sum(a, b):
    """Reduce"""
    return a + b


@task(returns=np.ndarray)
def _distributed_validate_X_predict(estimator, X):
    return estimator._validate_X_predict(X, check_input=True)


class BaseForest(six.with_metaclass(ABCMeta, BaseEnsemble)):
    """Base class for forests of trees.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=10,
                 estimator_params=tuple(),
                 bootstrap=True,
                 oob_score=False,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(BaseForest, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight

        self.n_features_ = None  # Initialized by fit(self, X, y)
        self.n_outputs_ = None   # Initialized by fit(self, X, y)
        self.estimators_ = []    # Kept distributed. Call compss_wait_on(e) to obtain an estimator e.

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

        results[:] = [compss_wait_on(res) for res in results]

        return np.array(results).T

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
                self._call_distributed_build_trees(
                    t, self.bootstrap, X, y, sample_weight, i, len(trees),
                    verbose=self.verbose, class_weight=self.class_weight)

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score(X, y)

        self._decapsulate_classes()

        return self

    @abstractmethod
    def _call_distributed_build_trees(self, *args, **kwargs):
        """Call _distributed_build_[regression|classifier]_trees"""

    def _decapsulate_classes(self):
        # Default implementation
        pass

    @abstractmethod
    def _set_oob_score(self, X, y):
        """Calculate out of bag predictions and score."""

    def _validate_y_class_weight(self, y):
        # Default implementation
        return y, None

    def _validate_X_predict(self, X):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        return _distributed_validate_X_predict(self.estimators_[0], X)

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


class RandomForestClassifier(six.with_metaclass(ABCMeta, BaseForest,
                                                ClassifierMixin)):
    """A random forest classifier.

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and use averaging to
    improve the predictive accuracy and control over-fitting. The sub-sample
    size is always the same as the original input sample size but the samples
    are drawn with replacement if bootstrap=True (default).
    """

    def __init__(self,
                 n_estimators=10,
                 bootstrap=True,
                 oob_score=False,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None):
        super(RandomForestClassifier, self).__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

    def _call_distributed_build_trees(self, *args, **kwargs):
        _distributed_build_classifier_trees(*args, **kwargs)

    def _decapsulate_classes(self):
        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

    def _set_oob_score(self, X, y):
        """Compute out-of-bag score"""
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_classes_ = self.n_classes_
        n_samples = y.shape[0]

        oob_decision_function = []
        oob_score = 0.0
        predictions = []

        for k in range(self.n_outputs_):
            predictions.append(np.zeros((n_samples, n_classes_[k])))

        p_estimator_array = []
        unsampled_indices_array = []

        for estimator in self.estimators_:
            p_estimator, unsampled_indices = _distributed_set_oob_score_classifier(X, estimator, n_samples, self.n_outputs_)
            p_estimator_array.append(p_estimator)
            unsampled_indices_array.append(unsampled_indices)

        for i in range(len(p_estimator_array)):
            p_estimator_array[i] = compss_wait_on(p_estimator_array[i])
            unsampled_indices_array[i] = compss_wait_on(unsampled_indices_array[i])

        for i in range(len(p_estimator_array)):
            for k in range(self.n_outputs_):
                predictions[k][unsampled_indices_array[i], :] += p_estimator_array[i][k]

        for k in range(self.n_outputs_):
            if (predictions[k].sum(axis=1) == 0).any():
                warn("Some inputs do not have OOB scores. "
                     "This probably means too few trees were used "
                     "to compute any reliable oob estimates.")
            decision = (predictions[k] /
                        predictions[k].sum(axis=1)[:, np.newaxis])
            oob_decision_function.append(decision)
            oob_score += np.mean(y[:, k] ==
                                 np.argmax(predictions[k], axis=1), axis=0)

        if self.n_outputs_ == 1:
            self.oob_decision_function_ = oob_decision_function[0]
        else:
            self.oob_decision_function_ = oob_decision_function

        self.oob_score_ = oob_score / self.n_outputs_

    def _validate_y_class_weight(self, y):
        check_classification_targets(y)

        y = np.copy(y)
        expanded_class_weight = None

        if self.class_weight is not None:
            y_original = np.copy(y)

        self.classes_ = []
        self.n_classes_ = []

        y_store_unique_indices = np.zeros(y.shape, dtype=np.int)
        for k in range(self.n_outputs_):
            classes_k, y_store_unique_indices[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
        y = y_store_unique_indices

        if self.class_weight is not None:
            valid_presets = ('balanced', 'balanced_subsample')
            if isinstance(self.class_weight, six.string_types):
                if self.class_weight not in valid_presets:
                    raise ValueError('Valid presets for class_weight include '
                                     '"balanced" and "balanced_subsample". Given "%s".'
                                     % self.class_weight)
                if self.warm_start:
                    warn('class_weight presets "balanced" or "balanced_subsample" are '
                         'not recommended for warm_start if the fitted data '
                         'differs from the full dataset. In order to use '
                         '"balanced" weights, use compute_class_weight("balanced", '
                         'classes, y). In place of y you can use a large '
                         'enough sample of the full training set target to '
                         'properly estimate the class frequency '
                         'distributions. Pass the resulting weights as the '
                         'class_weight parameter.')

            if (self.class_weight != 'balanced_subsample' or not self.bootstrap):
                if self.class_weight == "balanced_subsample":
                    class_weight = "balanced"
                else:
                    class_weight = self.class_weight
                expanded_class_weight = compute_sample_weight(class_weight, y_original)

        return y, expanded_class_weight

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            n_samples = proba[0].shape[0]
            predictions = np.zeros((n_samples, self.n_outputs_))

            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(np.argmax(proba[k],
                                                                    axis=1),
                                                          axis=0)

            return predictions

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest. The
        class probability of a single tree is the fraction of samples of the same
        class in a leaf.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        check_is_fitted(self, 'estimators_')
        # Check data
        X_np = self._validate_X_predict(X)

        # Sum the proba for every tree.
        predicts_reduction = []
        for e in self.estimators_:
            predicts_reduction.append(_distributed_predict_proba(e, X_np))

        while len(predicts_reduction) > 1:
            length = len(predicts_reduction)
            for i in range(length // 2):
                predicts_reduction[i] = _distributed_sum(predicts_reduction[i], predicts_reduction[-(i+1)])
            predicts_reduction = predicts_reduction[: length // 2 + length % 2]

        all_proba = compss_wait_on(predicts_reduction[0])

        for proba in all_proba:
            proba /= len(self.estimators_)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the log of the mean predicted class probabilities of the trees in the
        forest.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba


class RandomForestRegressor(six.with_metaclass(ABCMeta, BaseForest, RegressorMixin)):
    """A random forest regressor.

    A random forest is a meta estimator that fits a number of classifying decision
    trees on various sub-samples of the dataset and use averaging to improve the
    predictive accuracy and control over-fitting. The sub-sample size is always the
    same as the original input sample size but the samples are drawn with
    replacement if bootstrap=True (default).
    """

    def __init__(self,
                 n_estimators=10,
                 bootstrap=True,
                 oob_score=False,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None):
        estimator_params = ["criterion", "max_depth", "min_samples_split",
                            "min_samples_leaf", "min_weight_fraction_leaf",
                            "max_features", "max_leaf_nodes",
                            "min_impurity_decrease", "min_impurity_split",
                            "random_state"]
        estimator_params.append
        super(RandomForestRegressor, self).__init__(
            base_estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

    def _call_distributed_build_trees(self, *args, **kwargs):
        _distributed_build_regression_trees(*args, **kwargs)

    def _set_oob_score(self, X, y):
        """Compute out-of-bag scores"""
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_samples = y.shape[0]

        p_estimator_array = []
        unsampled_indices_array = []

        predictions = np.zeros((n_samples, self.n_outputs_))
        n_predictions = np.zeros((n_samples, self.n_outputs_))

        for estimator in self.estimators_:
            p_estimator, unsampled_indices = _distributed_set_oob_score_regressor(X, estimator, n_samples, self.n_outputs_)
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
            self.oob_score_ += r2_score(y[:, k],
                                        predictions[:, k])

        self.oob_score_ /= self.n_outputs_

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
        # Check data
        X_np = self._validate_X_predict(X)

        # Sum the prediction for every tree.
        predicts_reduction = []
        for e in self.estimators_:
            predicts_reduction.append(_distributed_predict(e, X_np))

        while len(predicts_reduction) > 1:
            length = len(predicts_reduction)
            for i in range(length // 2):
                predicts_reduction[i] = _distributed_sum(predicts_reduction[i], predicts_reduction[-(i+1)])
            predicts_reduction = predicts_reduction[: length // 2 + length % 2]

        y_hat = compss_wait_on(predicts_reduction[0])

        y_hat /= len(self.estimators_)

        return y_hat
