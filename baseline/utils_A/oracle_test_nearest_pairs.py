"""Oracle target coverage with explicit neighboring support views.

Half of the budget is chosen by the test-pose facility objective.  Each anchor
then receives its nearest still-unselected training-pose neighbor, increasing
input-to-input overlap while retaining target-pose coverage.
"""

import numpy as np

from .oracle_test_nearest import _objective, _pose_distance_matrix
from .pose_fps import _pairwise_pose_distances


def select(train_indices, count=16, *, order=None, test_indices=None, poses=None,
           **_kwargs):
    if count < 1:
        raise ValueError("context count must be positive")
    if count >= len(train_indices):
        return list(range(len(train_indices)))
    if order is None or test_indices is None or poses is None or not test_indices:
        raise ValueError(
            "oracle_test_nearest_pairs needs order, test_indices, and poses")

    train_poses = [poses[order[index]] for index in train_indices]
    test_poses = [poses[order[index]] for index in test_indices]
    target_distances = _pose_distance_matrix(train_poses, test_poses)
    pair_distances = _pairwise_pose_distances(train_poses)
    anchor_count = max(1, count // 2)
    selected = []
    nearest_target = np.full(len(test_indices), np.inf, dtype=np.float64)
    remaining = set(range(len(train_indices)))

    for _ in range(anchor_count):
        anchor = min(
            remaining,
            key=lambda candidate: (
                _objective(np.minimum(nearest_target, target_distances[candidate])),
                candidate,
            ),
        )
        selected.append(anchor)
        remaining.remove(anchor)
        nearest_target = np.minimum(nearest_target, target_distances[anchor])

    for anchor in list(selected):
        if len(selected) >= count or not remaining:
            break
        support = min(remaining, key=lambda candidate: (pair_distances[anchor, candidate], candidate))
        selected.append(support)
        remaining.remove(support)
        nearest_target = np.minimum(nearest_target, target_distances[support])

    while len(selected) < count and remaining:
        candidate = min(
            remaining,
            key=lambda value: (
                _objective(np.minimum(nearest_target, target_distances[value])), value),
        )
        selected.append(candidate)
        remaining.remove(candidate)
        nearest_target = np.minimum(nearest_target, target_distances[candidate])

    print(
        "[A:oracle_test_nearest_pairs] "
        f"anchors={anchor_count}, mean/max target distance="
        f"{nearest_target.mean():.5f}/{nearest_target.max():.5f}",
        flush=True,
    )
    return sorted(selected)
