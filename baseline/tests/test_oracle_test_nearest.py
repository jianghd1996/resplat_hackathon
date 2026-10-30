import unittest

import numpy as np

import oracle_test_nearest


def pose_at(x):
    rotation = np.eye(3, dtype=np.float64)
    center = np.array([float(x), 0.0, 0.0])
    translation = -(rotation @ center)
    return rotation, translation


class OracleTestNearestTests(unittest.TestCase):
    def test_selects_only_training_views_and_covers_test_positions(self):
        order = [f"{index}.png" for index in range(9)]
        poses = {name: pose_at(index) for index, name in enumerate(order)}
        test_indices = [0, 4, 8]
        train_indices = [1, 2, 3, 5, 6, 7]

        selected_positions = oracle_test_nearest.select(
            train_indices, 3, order=order, test_indices=test_indices, poses=poses)
        selected_indices = [train_indices[position] for position in selected_positions]

        self.assertEqual(len(selected_indices), 3)
        self.assertTrue(set(selected_indices).isdisjoint(test_indices))
        for test_index in test_indices:
            self.assertLessEqual(min(abs(test_index - value) for value in selected_indices), 1)

    def test_requires_test_pose_context(self):
        with self.assertRaises(ValueError):
            oracle_test_nearest.select([0, 1, 2], 2)


if __name__ == "__main__":
    unittest.main()
