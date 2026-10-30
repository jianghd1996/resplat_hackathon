"""Oracle A: choose training views that best cover held-out test camera poses.

This strategy never selects a test image, but it does inspect test camera poses.
It is therefore intended as a diagnostic upper bound unless target poses are
explicitly available and allowed by the evaluation protocol.

The distance between a training camera and a test camera combines normalized
camera-center distance with relative rotation angle.  Greedy facility-location
selection minimizes mean test-pose distance while also penalizing the worst
covered test pose.
"""

import numpy as np


def _camera_center(rotation, translation):
    return -(rotation.T @ translation)


def _relative_rotation_angle(left, right):
    relative = left @ right.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _pose_distance_matrix(train_poses, test_poses, rotation_weight=0.5):
    train_centers = np.stack([_camera_center(r, t) for r, t in train_poses])
    test_centers = np.stack([_camera_center(r, t) for r, t in test_poses])
    all_centers = np.vstack((train_centers, test_centers))
    scene_center = np.median(all_centers, axis=0)
    scene_scale = np.median(np.linalg.norm(all_centers - scene_center, axis=1))
    scene_scale = max(float(scene_scale), 1e-8)

    translation = np.linalg.norm(
        train_centers[:, None, :] - test_centers[None, :, :], axis=-1) / scene_scale
    rotation = np.empty_like(translation)
    for train_index, (train_rotation, _) in enumerate(train_poses):
        for test_index, (test_rotation, _) in enumerate(test_poses):
            rotation[train_index, test_index] = (
                _relative_rotation_angle(train_rotation, test_rotation) / np.pi)
    return translation + rotation_weight * rotation


def _objective(nearest_distances, worst_weight=0.25):
    return float(nearest_distances.mean() + worst_weight * nearest_distances.max())


def select(train_indices, count=16, *, order=None, test_indices=None, poses=None,
           **_kwargs):
    """Return positions in ``train_indices`` selected near the test poses."""
    if count < 1:
        raise ValueError("context count must be positive")
    if count >= len(train_indices):
        return list(range(len(train_indices)))
    if order is None or test_indices is None or poses is None:
        raise ValueError(
            "oracle_test_nearest needs order, test_indices, and poses from preprocess")
    if not test_indices:
        raise ValueError("oracle_test_nearest needs at least one held-out test view")

    train_poses = [poses[order[index]] for index in train_indices]
    test_poses = [poses[order[index]] for index in test_indices]
    distances = _pose_distance_matrix(train_poses, test_poses)

    selected = []
    nearest = np.full(len(test_indices), np.inf, dtype=np.float64)
    remaining = set(range(len(train_indices)))
    while len(selected) < count:
        best_candidate = min(
            remaining,
            key=lambda candidate: (
                _objective(np.minimum(nearest, distances[candidate])), candidate),
        )
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        nearest = np.minimum(nearest, distances[best_candidate])

    baseline_positions = np.linspace(
        0, len(train_indices) - 1, count, dtype=int).tolist()
    baseline_nearest = distances[baseline_positions].min(axis=0)
    print(
        "[A:oracle_test_nearest] test-pose objective "
        f"uniform={_objective(baseline_nearest):.5f} -> "
        f"oracle={_objective(nearest):.5f}; "
        f"mean/max={nearest.mean():.5f}/{nearest.max():.5f}",
        flush=True,
    )
    # Preserve the dataset's canonical ordering for ReSplat.
    return sorted(selected)
