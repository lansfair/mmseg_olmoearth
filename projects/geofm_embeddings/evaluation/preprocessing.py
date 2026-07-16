from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def apply_track(
    values: np.ndarray,
    train_mask: np.ndarray,
    pca_dim: int | None,
    seed: int,
) -> np.ndarray:
    if pca_dim is None or pca_dim >= values.shape[1]:
        return values.copy()
    max_dim = min(int(train_mask.sum()), values.shape[1])
    if pca_dim >= max_dim:
        raise ValueError(f"PCA dimension {pca_dim} must be below train rank bound {max_dim}")
    pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=seed)
    pca.fit(values[train_mask])
    return pca.transform(values).astype(np.float32)


def l2(values: np.ndarray) -> np.ndarray:
    return normalize(values, norm="l2", axis=1, copy=True).astype(np.float32)
