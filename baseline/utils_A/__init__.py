"""Module A context-view strategies.

Each strategy exposes select(train_indices, count), returning positions in the
ordered training-view list. Pose loading and image normalization stay shared.
"""

from .uniform16 import select as select_uniform16
from .fps import select as select_fps

STRATEGIES = {"uniform16": select_uniform16, "fps": select_fps}


def select_context(strategy, train_indices, count, centers=None):
    try:
        selector = STRATEGIES[strategy]
    except KeyError as exc:
        raise ValueError(f"unknown module A strategy: {strategy}") from exc
    return selector(train_indices, count, centers=centers)
