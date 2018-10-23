import unittest
from test_split import test_split
import numpy as np


class TestTestSplit(unittest.TestCase):

    def test_1(self):
        # Testing impossible split
        sample = np.array([0, 1, 2, 3])
        y_s = np.array([1, 1, 1, 1])
        feature = np.array([0.9, 0.3, 0.2, 0.4])
        n_classes = 3
        best_score, best_value = test_split(sample, y_s, feature, n_classes)
        self.assertEquals(best_value, np.inf)

    def test_2(self):
        sample = np.array([9, 8, 7, 6, 10])
        y_s = np.array([1, 2, 1, 2, 2])
        feature = np.array([0, 0, 0, 0, 0, 0, 0.9, 0.7, 0.8, 0.6, 0.5])
        n_classes = 3
        best_score, best_value = test_split(sample, y_s, feature, n_classes)
        self.assertAlmostEqual(best_value, 0.75)

    def test_3(self):
        sample = np.array([0, 1, 2, 3])
        y_s = np.array([0, 1, 0, 0])
        feature = np.array([-2.45, -2.44, -2.4, -2.4])
        n_classes = 3
        best_score, best_value = test_split(sample, y_s, feature, n_classes)
        self.assertAlmostEqual(best_value, -2.42)

    def test_4(self):
        # Testing avoiding splits by repeated values
        sample = np.arange(11)
        y_s = np.array([0, 1, 0, 0, 2, 1, 0, 1, 0, 2, 2])
        feature = np.array([0.018, 0.018,  0.018,  0.018,  0.018, 0.0018, 0.000018, 0.018, 0.018, 0.018, 0.018])
        n_classes = 3
        best_score, best_value = test_split(sample, y_s, feature, n_classes)
        self.assertAlmostEqual(best_value, 0.000909)


if __name__ == '__main__':
    unittest.main()
