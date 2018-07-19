from __future__ import division
from collections import Counter

from pycompss.api.api import compss_wait_on

from decision_tree import DecisionTreeClassifier
from decision_tree import get_features_file, get_feature_task
from decision_tree import get_y

import numpy as np
import warnings


class RandomForestClassifier:
    def __init__(self,
                 path_in,
                 n_instances,
                 n_features,
                 path_out,
                 n_estimators=10,
                 max_depth=None,
                 distr_depth=None):
        self.path_in = path_in
        self.n_instances = n_instances
        self.n_features = n_features
        self.path_out = path_out
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.distr_depth = distr_depth

        self.y = None
        self.y_codes = None
        self.n_classes = None
        self.trees = []

    def fit(self):
        """
        Fits the RandomForestClassifier.
        """
        features = []
        features_file = get_features_file(self.path_in)
        for i in range(self.n_features):
            features.append(get_feature_task(features_file, i))
        self.y, self.y_codes, self.n_classes = get_y(self.path_in)

        for i in range(self.n_estimators):
            tree = DecisionTreeClassifier(self.path_in, self.n_instances, self.n_features,
                                self.path_out, 'tree_' + str(i), self.max_depth, self.distr_depth)
            tree.features = features
            tree.y_codes = self.y_codes
            tree.n_classes = self.n_classes
            self.trees.append(tree)

        for tree in self.trees:
            tree.fit()

        self.y, self.y_codes, self.n_classes = compss_wait_on(self.y, self.y_codes, self.n_classes)

    def predict_probabilities(self):
        """ Predicts class probabilities by class code using a fitted forest and returns a 1D or 2D array. """
        try:
            x_test = np.load(self.path_in + 'x_test.npy', allow_pickle=False)
        except IOError:
            warnings.warn('No test data found in the input path.')
            return

        return sum(tree.predict_probabilities(x_test) for tree in self.trees) / len(self.trees)

    def predict(self, file_name='x_test.npy', soft_voting=False):
        """ Predicts classes using a fitted forest and returns an integer or an array. """
        try:
            x_test = np.load(self.path_in + file_name, allow_pickle=False)
        except IOError:
            warnings.warn('No test data found in the input path.')
            return

        if soft_voting:
            probabilities = self.predict_probabilities(x_test)
            return self.y.take(np.argmax(probabilities, axis=1), axis=0)

        if len(x_test.shape) == 1:
            predicted = Counter(tree.predict(x_test) for tree in self.trees).most_common(1)[0][0]
            return self.y.categories[predicted]  # Convert code to real value
        elif len(x_test.shape) == 2:
            my_array = np.empty((len(self.trees), len(x_test)), np.int64)
            for i, tree in enumerate(self.trees):
                my_array[i, :] = tree.predict(x_test)
            predicted = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), 0, my_array)
            return self.y.categories[predicted]  # Convert codes to real values
        else:
            raise ValueError
