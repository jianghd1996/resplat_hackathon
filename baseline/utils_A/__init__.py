"""Module A context-view strategies.

Each strategy exposes select(train_indices, count), returning positions in the
ordered training-view list. Pose loading and image normalization stay shared.
"""

from .oracle_test_nearest import select as select_oracle_test_nearest
from .uniform16 import select as select_uniform16

STRATEGIES = {
    "oracle_test_nearest": select_oracle_test_nearest,
    "uniform16": select_uniform16,
}


def select_context(strategy, train_indices, count, **kwargs):
    try:
        selector = STRATEGIES[strategy]
    except KeyError as exc:
        raise ValueError(f"unknown module A strategy: {strategy}") from exc
    return selector(train_indices, count, **kwargs)
