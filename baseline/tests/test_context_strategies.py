import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils_A import STRATEGIES, select_context


def circle_pose(angle):
    center = np.array([np.cos(angle), np.sin(angle), 0.15 * np.sin(2 * angle)])
    forward = -center / np.linalg.norm(center)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    translation = -(rotation @ center)
    return rotation, translation


class ContextStrategyTests(unittest.TestCase):
    def test_all_strategies_return_unique_training_positions(self):
        order = [f"{index:03d}.png" for index in range(48)]
        poses = {
            name: circle_pose(2 * np.pi * index / len(order))
            for index, name in enumerate(order)
        }
        test_indices = [index for index in range(len(order)) if index % 8 == 0]
        train_indices = [index for index in range(len(order)) if index % 8 != 0]
        expected = {
            "uniform16", "random16", "pose_fps", "orbit_fps",
            "oracle_test_nearest", "oracle_test_nearest_pairs",
        }
        self.assertTrue(expected.issubset(STRATEGIES))
        for strategy in sorted(expected):
            with self.subTest(strategy=strategy):
                selected = select_context(
                    strategy, train_indices, 16,
                    order=order, test_indices=test_indices, poses=poses)
                self.assertEqual(len(selected), 16)
                self.assertEqual(len(set(selected)), 16)
                self.assertTrue(all(0 <= value < len(train_indices) for value in selected))


if __name__ == "__main__":
    unittest.main()
