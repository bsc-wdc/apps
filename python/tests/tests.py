import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class ImportTests(unittest.TestCase):

    def test_import_cascadecsvm(self):
        from pycompss_lib.ml.classification import CascadeSVM

    def test_import_random_forest(self):
        from pycompss_lib.ml.classification import RandomForestClassifier

    def test_import_kmeans(self):
        from pycompss_lib.ml.clustering import Kmeans

    def test_import_dbscan(self):
        from pycompss_lib.ml.clustering import DBSCAN

    def test_import_pca(self):
        from pycompss_lib.ml.analysis import pca

    def test_import_cholesky(self):
        from pycompss_lib.math.linalg import cholesky

    def test_import_matmul(self):
        from pycompss_lib.math.linalg import matmul

    def test_import_qr(self):
        from pycompss_lib.math.linalg import qr

    def test_import_max_norm(self):
        from pycompss_lib.algorithms import max_norm

    def test_import_terasort(self):
        from pycompss_lib.algorithms import terasort

    def test_import_sort(self):
        from pycompss_lib.algorithms import sort

    def test_import_sort_by_key(self):
        from pycompss_lib.algorithms import sort_by_key


class ResultsTest(unittest.TestCase):
    #import pycompss.interactive as ipycompss
    #ipycompss.start(graph=False, trace=False, debug=False)
    #ipycompss.start(graph=True, trace=True, debug=True, project_xml='../project.xml', resources_xml='../resources.xml')

    def test_cascadecsvm(self):
        from pycompss_lib.ml.classification import CascadeSVM

    def test_import_max_norm(self):
        from pycompss_lib.algorithms import max_norm
        
        points, dimensions, fragments, seed = 16000, 3, 16, 666
        expected_output = 17137.68885398087

        result = max_norm(points, dimensions, fragments, seed)

        self.assertEqual(result, expected_output)


def main():
    unittest.main()

if __name__ == '__main__':
    main()