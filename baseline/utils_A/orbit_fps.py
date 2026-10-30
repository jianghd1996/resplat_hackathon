"""Farthest-point sampling of camera-center directions around the scene."""

import numpy as np

from .oracle_test_nearest import _camera_center


def select(train_indices, count=16, *, order=None, poses=None, **_kwargs):
    if count < 1:
        raise ValueError("context count must be positive")
    if count >= len(train_indices):
        return list(range(len(train_indices)))
    if order is None or poses is None:
        raise ValueError("orbit_fps needs order and poses from preprocess")
    all_centers = np.stack([
        _camera_center(*poses[name]) for name in order
    ])
    scene_center = np.median(all_centers, axis=0)
    train_centers = np.stack([
        _camera_center(*poses[order[index]]) for index in train_indices
    ])
    directions = train_centers - scene_center
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    distances = np.arccos(np.clip(directions @ directions.T, -1.0, 1.0))
    selected = [int(np.argmax(distances.mean(axis=1)))]
    nearest = distances[selected[0]].copy()
    while len(selected) < count:
        nearest[selected] = -1.0
        candidate = int(np.argmax(nearest))
        selected.append(candidate)
        nearest = np.minimum(nearest, distances[candidate])
    return sorted(selected)
