"""Select globally diverse training cameras with farthest-point sampling."""

import numpy as np

from .oracle_test_nearest import _camera_center, _relative_rotation_angle


def _pairwise_pose_distances(poses):
    centers = np.stack([_camera_center(rotation, translation)
                        for rotation, translation in poses])
    center = np.median(centers, axis=0)
    scale = max(float(np.median(np.linalg.norm(centers - center, axis=1))), 1e-8)
    translation = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1) / scale
    rotation = np.zeros_like(translation)
    for i, (left, _) in enumerate(poses):
        for j in range(i + 1, len(poses)):
            value = _relative_rotation_angle(left, poses[j][0]) / np.pi
            rotation[i, j] = rotation[j, i] = value
    return translation + 0.5 * rotation


def select(train_indices, count=16, *, order=None, poses=None, **_kwargs):
    if count < 1:
        raise ValueError("context count must be positive")
    if count >= len(train_indices):
        return list(range(len(train_indices)))
    if order is None or poses is None:
        raise ValueError("pose_fps needs order and poses from preprocess")
    train_poses = [poses[order[index]] for index in train_indices]
    distances = _pairwise_pose_distances(train_poses)
    # A max-mean starting point is deterministic and avoids privileging filename order.
    selected = [int(np.argmax(distances.mean(axis=1)))]
    nearest = distances[selected[0]].copy()
    while len(selected) < count:
        nearest[selected] = -1.0
        candidate = int(np.argmax(nearest))
        selected.append(candidate)
        nearest = np.minimum(nearest, distances[candidate])
    return sorted(selected)
