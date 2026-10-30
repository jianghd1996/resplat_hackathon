"""Deterministic random context selection for an A/B control."""

import hashlib

import numpy as np


def select(train_indices, count=16, *, order=None, **_kwargs):
    if count < 1:
        raise ValueError("context count must be positive")
    if count >= len(train_indices):
        return list(range(len(train_indices)))
    identity = "|".join(order or map(str, train_indices)).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
    generator = np.random.default_rng(seed)
    return sorted(int(value) for value in generator.choice(
        len(train_indices), size=count, replace=False))
