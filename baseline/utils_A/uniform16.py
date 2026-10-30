"""Baseline A: evenly sample up to 16 ordered training views."""

import numpy as np


def select(train_indices, count=16, **_kwargs):
    if count < 1:
        raise ValueError("context count must be positive")
    if count >= len(train_indices):
        return list(range(len(train_indices)))
    return sorted(set(int(i) for i in np.linspace(
        0, len(train_indices) - 1, count, dtype=int)))
