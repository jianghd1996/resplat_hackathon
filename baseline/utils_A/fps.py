"""Select spatially spread training cameras with deterministic farthest points."""

import numpy as np


def select(train_indices, count=16, centers=None, directions=None, image_paths=None,
           track_sets=None):
    if count < 1:
        raise ValueError("context count must be positive")
    if centers is None:
        raise ValueError("FPS needs camera centers for the training views")
    points = np.asarray(centers, dtype=np.float64)
    if points.shape != (len(train_indices), 3) or not np.isfinite(points).all():
        raise ValueError("camera centers must have shape (number of training views, 3) and be finite")
    if count >= len(train_indices):
        return list(range(len(train_indices)))

    first = int(np.argmax(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))
    chosen = [first]
    nearest_sq = np.sum((points - points[first]) ** 2, axis=1)
    nearest_sq[first] = -np.inf
    while len(chosen) < count:
        next_index = int(np.argmax(nearest_sq))
        chosen.append(next_index)
        nearest_sq = np.minimum(nearest_sq, np.sum((points - points[next_index]) ** 2, axis=1))
        nearest_sq[chosen] = -np.inf
    return sorted(chosen)
