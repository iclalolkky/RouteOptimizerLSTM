import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from route_optimizer import build_target_sizes, split_into_balanced_clusters


class RouteOptimizerBalancingTests(unittest.TestCase):
    def test_build_target_sizes_spreads_stops_evenly(self):
        self.assertEqual(build_target_sizes(12, 5), [3, 3, 2, 2, 2])

    def test_balanced_clusters_keep_all_five_trucks_in_play(self):
        positions = [0, 8, 10, 12, 18, 20, 22, 28, 30, 32, 38]
        distance_matrix = np.abs(np.subtract.outer(positions, positions))

        clusters = split_into_balanced_clusters(distance_matrix, num_vehicles=5)
        sizes = [len(cluster) for cluster in clusters]

        self.assertEqual(len(clusters), 5)
        self.assertTrue(all(size > 0 for size in sizes))
        self.assertLessEqual(max(sizes) - min(sizes), 1)


if __name__ == '__main__':
    unittest.main()
