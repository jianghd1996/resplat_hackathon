"""Module A context-view strategies.

Each strategy exposes select(train_indices, count), returning positions in the
ordered training-view list. Pose loading and image normalization stay shared.
"""

from .uniform16 import select as select_uniform16
from .fps import select as select_fps
from .quality_filter import select as select_quality_filter
from .coverage_local import select as select_coverage_local
from .covisibility_guarded import select as select_covisibility_guarded

STRATEGIES = {"uniform16": select_uniform16, "fps": select_fps,
              "quality_filter": select_quality_filter,
              "coverage_local": select_coverage_local,
              "covisibility_guarded": select_covisibility_guarded}


def select_context(strategy, train_indices, count, centers=None, directions=None,
                   image_paths=None, track_sets=None):
    try:
        selector = STRATEGIES[strategy]
    except KeyError as exc:
        raise ValueError(f"unknown module A strategy: {strategy}") from exc
    return selector(train_indices, count, centers=centers, directions=directions,
                    image_paths=image_paths, track_sets=track_sets)
