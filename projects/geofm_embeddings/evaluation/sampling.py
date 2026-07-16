from __future__ import annotations

import numpy as np


def balanced_per_split_indices(
    labels: np.ndarray,
    split: np.ndarray,
    maximum_per_class: int,
    seed: int,
) -> np.ndarray:
    if maximum_per_class < 1:
        raise ValueError("maximum_per_class must be positive")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for split_name in np.unique(split):
        split_indices = np.flatnonzero(split == split_name)
        for label in np.unique(labels[split_indices]):
            candidates = split_indices[labels[split_indices] == label]
            count = min(maximum_per_class, len(candidates))
            selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)
