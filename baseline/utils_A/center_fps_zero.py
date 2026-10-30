"""Farthest-point sampling on 3-D camera centers, starting from view zero.

This intentionally mirrors the legacy sampler: no rotation term, no scene-scale
normalization, and the first item in the ordered training-view list is fixed as
the initial sample.
"""

import numpy as np

from .oracle_test_nearest import _camera_center


def farthest_point_sampling(points, selected_num, start_with_zero=True):
    """Return sorted FPS indices for an N x 3 point array."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if selected_num < 1:
        raise ValueError("selected_num must be positive")
    if selected_num >= len(points):
        return list(range(len(points)))

    # The registered strategy always requests start_with_zero=True.  Keeping
    # the alternate path makes this helper reusable while remaining explicit.
    first_index = 0 if start_with_zero else int(np.random.default_rng().integers(len(points)))
    selected_index = [first_index]
    distance = np.sum(np.square(points - points[first_index]), axis=1)

    while len(selected_index) < selected_num:
        # Excluding already selected indices only changes degenerate duplicate-
        # center cases, where the legacy implementation could return duplicates.
        distance[selected_index] = -np.inf
        new_index = int(np.argmax(distance))
        new_distance = np.sum(np.square(points - points[new_index]), axis=1)
        distance = np.minimum(distance, new_distance)
        selected_index.append(new_index)

    return sorted(selected_index)


def select(train_indices, count=16, *, order=None, poses=None, **_kwargs):
    if order is None or poses is None:
        raise ValueError("center_fps_zero needs order and poses from preprocess")
    train_poses = [poses[order[index]] for index in train_indices]
    centers = np.stack([
        _camera_center(rotation, translation)
        for rotation, translation in train_poses
    ])
    return farthest_point_sampling(centers, count, start_with_zero=True)
