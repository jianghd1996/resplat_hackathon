"""Select dispersed boundary views together with nearby pose neighbors."""

import numpy as np


def select(train_indices, count=16, centers=None, directions=None, image_paths=None,
           track_sets=None):
    if count < 1:
        raise ValueError("context count must be positive")
    n = len(train_indices)
    if count >= n:
        return list(range(n))
    points = np.asarray(centers, dtype=np.float64)
    forward = np.asarray(directions, dtype=np.float64)
    if points.shape != (n, 3) or forward.shape != (n, 3):
        raise ValueError("coverage_local needs one center and forward direction per train view")
    if not np.isfinite(points).all() or not np.isfinite(forward).all():
        raise ValueError("camera geometry must be finite")
    lengths = np.linalg.norm(forward, axis=1)
    if np.any(lengths < 1e-9):
        raise ValueError("camera forward directions must be nonzero")
    forward = forward / lengths[:, None]

    span = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1e-9)
    spatial = np.linalg.norm(points[:, None] - points[None, :], axis=2) / span
    angular = np.sqrt(np.clip((1.0 - forward @ forward.T) / 2.0, 0.0, 1.0))
    distance = 0.6 * spatial + 0.4 * angular

    # Half the budget covers the boundary; the remaining views add local
    # overlap near each anchor. All ties follow ordered train-view indices.
    anchor_count = (count + 1) // 2
    first = int(np.argmax(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))
    anchors = [first]
    while len(anchors) < anchor_count:
        nearest = np.min(distance[:, anchors], axis=1)
        nearest[anchors] = -np.inf
        anchors.append(int(np.argmax(nearest)))
    chosen = set(anchors)
    for anchor in anchors:
        if len(chosen) == count:
            break
        available = (i for i in range(n) if i not in chosen)
        chosen.add(min(available, key=lambda i: (distance[anchor, i], i)))
    return sorted(chosen)
