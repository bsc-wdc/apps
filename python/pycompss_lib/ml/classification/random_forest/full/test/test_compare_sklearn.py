from __future__ import division
import unittest

from sklearn.datasets import load_iris
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier as SKDecisionTree
from sklearn.ensemble import RandomForestClassifier as SKRandomForest
from profile_decision_tree import DecisionTreeClassifier
from profile_forest import RandomForestClassifier


class TestCompareSKLearn(unittest.TestCase):

    def setUp(self):
        x, y = load_iris(return_X_y=True)
        self.x_train = x[::2]
        self.y_train = y[::2]
        self.x_test = x[1::2]
        self.y_test = y[1::2]
        np.save('x.npy', self.x_train)
        np.save('x_t.npy', self.x_train.T.copy(order='C'))
        np.savetxt('y.dat', self.y_train, fmt='%s')
        np.save('x_test.npy', self.x_test)

    def test_1(self):
        sk_tree = SKDecisionTree()
        sk_tree.fit(self.x_train, self.y_train)
        y_predicted_sk_tree = sk_tree.predict(self.x_test)
        accuracy_sk_tree = (np.count_nonzero(y_predicted_sk_tree == self.y_test)) / len(self.y_test)

        sk_forest = SKRandomForest()
        sk_forest.fit(self.x_train, self.y_train)
        y_predicted_sk_forest = sk_forest.predict(self.x_test)
        accuracy_sk_forest = (np.count_nonzero(y_predicted_sk_forest == self.y_test)) / len(self.y_test)

        tree = DecisionTreeClassifier(path_in=os.getcwd(), n_instances=75, n_features=4, path_out=os.getcwd(), name_out='test_tree')
        tree.fit()
        y_predicted_tree = tree.predict(self.x_test)
        accuracy_tree = (np.count_nonzero(y_predicted_tree == self.y_test)) / len(self.y_test)

        forest = RandomForestClassifier(os.getcwd(), 75, 4, os.getcwd())
        forest.fit()
        y_predicted_forest = forest.predict()
        accuracy_forest = np.count_nonzero(y_predicted_forest == [str(num) for num in self.y_test]) / len(self.y_test)

        print 'Iris dataset'
        print 'Accuracies (sk_tree, sk_forest, tree, forest):'
        print accuracy_sk_tree, accuracy_sk_forest, accuracy_tree, accuracy_forest
        self.assertGreaterEqual(accuracy_tree, 0.9)
        self.assertGreaterEqual(accuracy_forest, 0.9)


if __name__ == '__main__':
    unittest.main()
