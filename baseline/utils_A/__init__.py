"""Module A context-view strategies.

Each strategy exposes select(train_indices, count), returning positions in the
ordered training-view list. Pose loading and image normalization stay shared.
"""

from .oracle_test_nearest import select as select_oracle_test_nearest
from .oracle_test_nearest_pairs import select as select_oracle_test_nearest_pairs
from .orbit_fps import select as select_orbit_fps
from .pose_fps import select as select_pose_fps
from .random16 import select as select_random16
from .uniform16 import select as select_uniform16

STRATEGIES = {
    "oracle_test_nearest": select_oracle_test_nearest,
    "oracle_test_nearest_pairs": select_oracle_test_nearest_pairs,
    "orbit_fps": select_orbit_fps,
    "pose_fps": select_pose_fps,
    "random16": select_random16,
    "uniform16": select_uniform16,
}


def select_context(strategy, train_indices, count, **kwargs):
    try:
        selector = STRATEGIES[strategy]
    except KeyError as exc:
        raise ValueError(f"unknown module A strategy: {strategy}") from exc
    return selector(train_indices, count, **kwargs)
