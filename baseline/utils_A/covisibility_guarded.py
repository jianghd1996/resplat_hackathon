"""Improve uniform16 train-track support with at most two guarded swaps."""

import numpy as np

from .uniform16 import select as select_uniform16


def select(train_indices, count=16, centers=None, directions=None, image_paths=None,
           track_sets=None):
    uniform = select_uniform16(train_indices, count)
    n = len(train_indices)
    if len(uniform) == n or not track_sets or len(track_sets) != n:
        return uniform
    tracks = [set(map(int, values)) for values in track_sets]
    universe = sorted(set().union(*tracks))
    if not universe:
        return uniform
    points = np.asarray(centers, dtype=np.float64)
    if points.shape != (n, 3) or not np.isfinite(points).all():
        return uniform
    lookup = {key: i for i, key in enumerate(universe)}
    track_ids = [np.asarray([lookup[key] for key in values], dtype=int) for values in tracks]
    distance = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    distance /= max(float(np.linalg.norm(np.ptp(points, axis=0))), 1e-12)

    def statistics(ids):
        counts = np.zeros(len(universe), dtype=np.int16)
        for index in ids:
            counts[track_ids[index]] += 1
        visible = counts > 0
        support = [visible[ids_].mean() for ids_ in track_ids if len(ids_)]
        unselected = [i for i in range(n) if i not in ids]
        return counts, {
            "track_coverage": float(visible.mean()),
            "support_p10": float(np.quantile(support, 0.1)),
            "nearest_distance": float(distance[unselected][:, ids].min(1).mean()),
            "max_index_gap": int(np.diff(sorted(ids)).max()),
        }

    selected = uniform.copy()
    counts, baseline = statistics(selected)
    boundaries = np.asarray([(uniform[i] + uniform[i + 1]) / 2
                             for i in range(len(uniform) - 1)])
    bins = np.searchsorted(boundaries, np.arange(n), side="right")
    for _ in range(2):
        proposals = []
        for slot in range(1, len(selected) - 1):
            remaining = counts.copy()
            remaining[track_ids[selected[slot]]] -= 1
            lost = np.count_nonzero(counts) - np.count_nonzero(remaining)
            for candidate in np.flatnonzero(bins == slot):
                if candidate in selected:
                    continue
                gain = np.count_nonzero(remaining[track_ids[candidate]] == 0) - lost
                if gain > 0:
                    proposals.append((-int(gain), slot, int(candidate)))
        accepted = False
        for _, slot, candidate in sorted(proposals):
            trial = selected.copy()
            trial[slot] = candidate
            new_counts, stats = statistics(trial)
            if (stats["nearest_distance"] > 1.1 * baseline["nearest_distance"] or
                    stats["support_p10"] < baseline["support_p10"] or
                    stats["max_index_gap"] > 2 * baseline["max_index_gap"]):
                continue
            selected, counts, accepted = trial, new_counts, True
            break
        if not accepted:
            break
    return sorted(selected)
