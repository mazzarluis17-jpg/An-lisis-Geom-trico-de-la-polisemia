import unittest

import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness

from f2 import classical, graph, grid, paths
from geometria import coincidencia, concordancia, normalizar, vecinos


class Geometria(unittest.TestCase):
    def test_nonmutating_normalization(self):
        x = np.array([[3, 4], [1, 1]], dtype=np.float32)
        old = x.copy()
        np.testing.assert_allclose(np.linalg.norm(normalizar(x), axis=1), 1)
        np.testing.assert_array_equal(x, old)

    def test_tie_break_is_stable(self):
        rank = vecinos(np.array([[0, 0], [1, 0], [-1, 0]]))
        np.testing.assert_array_equal(rank[0], [1, 2])

    def test_complete_graph_matches_pca(self):
        x = np.random.default_rng(27).normal(size=(20, 8))
        d = cdist(x, x)
        rank = vecinos(x)
        full = graph(rank, len(x)-1)
        self.assertTrue(np.all(~graph(rank, 2) | full))
        geo, _ = paths(d, full)
        np.testing.assert_allclose(geo, d, atol=1e-12)
        _, z = classical(geo)
        pca = PCA(n_components=2, svd_solver='full').fit_transform(x)
        self.assertLess(procrustes(z, pca)[2], 1e-12)
        self.assertEqual(coincidencia(vecinos(z), vecinos(pca), 5), 1)
        self.assertEqual(concordancia(vecinos(z), vecinos(pca), 5), 1)

    def test_disconnected_graph_retains_infinite_paths(self):
        x = np.array([[0], [1], [10], [11]])
        d = cdist(x, x)
        geo, _ = paths(d, graph(vecinos(x), 1))
        self.assertTrue(np.isinf(geo[0, 2]))
        self.assertEqual(geo[0, 1], 1)

    def test_grid_respects_connectivity_floor(self):
        proposed, actual = grid(24, 12)
        self.assertEqual(proposed['prop0.40'], 10)
        self.assertEqual(actual['prop0.40'], 12)
        self.assertEqual(actual['min+4'], 16)
        self.assertEqual(len(actual), 11)

    def test_trustworthiness_small_group(self):
        x = np.random.default_rng(27).normal(size=(20, 8))
        self.assertEqual(trustworthiness(x, x, n_neighbors=5), 1)
        with self.assertRaises(ValueError):
            trustworthiness(x, x, n_neighbors=10)


if __name__ == '__main__':
    unittest.main()
