"""Use guarded covisibility swaps only when train-track gains cross-validate."""

import numpy as np

from .covisibility_guarded import select as select_guarded
from .uniform16 import select as select_uniform16


NUM_FOLDS = 5
MIN_POSITIVE_FOLDS = 4
MIN_MEDIAN_COVERAGE_GAIN = 0.01
MIN_MEDIAN_SUPPORT_GAIN = 0.0
_MASK64 = (1 << 64) - 1


def _proposal_track(point_id, fold):
    value = (int(point_id) * 0x9E3779B185EBCA87 +
             fold * 0xC2B2AE3D27D4EB4F) & _MASK64
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _MASK64
    value ^= value >> 27
    return (value & 1) == 0


def _coverage(ids, tracks):
    universe = set().union(*tracks) if tracks else set()
    if not universe:
        return 0.0, 0.0
    visible = set().union(*(tracks[index] for index in ids))
    support = [len(values & visible) / len(values) for values in tracks if values]
    p10 = float(np.quantile(support, 0.1)) if support else 0.0
    return len(visible) / len(universe), p10


def diagnose(train_indices, count=16, centers=None, track_sets=None):
    uniform = select_uniform16(train_indices, count)
    n = len(train_indices)
    if not track_sets or len(track_sets) != n or not any(track_sets):
        return {"selected": uniform, "candidate": uniform,
                "accepted": False, "reason": "no_train_tracks", "folds": []}
    tracks = [set(map(int, values)) for values in track_sets]
    candidate = select_guarded(train_indices, count, centers=centers,
                               track_sets=tracks)
    if candidate == uniform:
        return {"selected": uniform, "candidate": candidate,
                "accepted": False, "reason": "no_guarded_swap", "folds": []}

    folds = []
    for fold in range(NUM_FOLDS):
        proposal = [{point for point in values if _proposal_track(point, fold)}
                    for values in tracks]
        heldout = [values - proposed for values, proposed in zip(tracks, proposal)]
        fold_candidate = select_guarded(train_indices, count, centers=centers,
                                        track_sets=proposal)
        uniform_coverage, uniform_support = _coverage(uniform, heldout)
        candidate_coverage, candidate_support = _coverage(fold_candidate, heldout)
        folds.append({
            "coverage_gain": candidate_coverage - uniform_coverage,
            "support_p10_gain": candidate_support - uniform_support,
            "changed_slots": sum(a != b for a, b in zip(fold_candidate, uniform)),
            "matches_full_candidate": fold_candidate == candidate,
        })
    coverage = np.asarray([item["coverage_gain"] for item in folds])
    support = np.asarray([item["support_p10_gain"] for item in folds])
    positive = int((coverage > 0).sum())
    median_coverage = float(np.median(coverage))
    median_support = float(np.median(support))
    accepted = (positive >= MIN_POSITIVE_FOLDS and
                median_coverage >= MIN_MEDIAN_COVERAGE_GAIN and
                median_support >= MIN_MEDIAN_SUPPORT_GAIN)
    return {
        "selected": candidate if accepted else uniform,
        "candidate": candidate,
        "accepted": accepted,
        "reason": "stable_material_track_gain" if accepted else "insufficient_track_gain",
        "positive_coverage_folds": positive,
        "median_coverage_gain": median_coverage,
        "median_support_p10_gain": median_support,
        "folds": folds,
    }


def select(train_indices, count=16, centers=None, directions=None, image_paths=None,
           track_sets=None):
    return diagnose(train_indices, count, centers=centers,
                    track_sets=track_sets)["selected"]
