import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_context_selections as comparison


class ContextSelectionComparisonTests(unittest.TestCase):
    def test_visibility_and_overlap_graph(self):
        camera = {
            "fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 1.0,
            "width": 2, "height": 2,
        }
        identity = {
            "R": np.eye(3), "t": np.zeros(3),
            "center": np.zeros(3), "name": "identity.png",
        }
        points = np.array([
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 1.0],
            [3.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ])

        visibility = comparison.visibility_matrix(points, [identity, identity], camera, 1e-4)
        np.testing.assert_array_equal(visibility[0], [True, True, False, False])
        jaccard = comparison.pairwise_jaccard(visibility)
        self.assertAlmostEqual(jaccard[0, 1], 1.0)
        graph = comparison.graph_stats(jaccard, 0.15)
        self.assertEqual(graph["num_components"], 1)
        self.assertEqual(graph["min_degree"], 1)

    def test_rotation_angles(self):
        identity = np.eye(3)[None]
        half_turn = np.diag([-1.0, -1.0, 1.0])[None]
        angles = comparison.rotation_angle_matrix(identity, half_turn)
        self.assertAlmostEqual(angles[0, 0], 180.0)


if __name__ == "__main__":
    unittest.main()
