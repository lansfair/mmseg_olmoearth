from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors


def participation_ratio(values: np.ndarray) -> float:
    centered = values.astype(np.float64) - values.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = singular_values**2 / max(len(values) - 1, 1)
    denominator = np.sum(eigenvalues**2)
    return float(np.sum(eigenvalues) ** 2 / denominator) if denominator > 0 else np.nan


def twonn(values: np.ndarray) -> float:
    distances = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(values).kneighbors(values)[0]
    r1 = np.maximum(distances[:, 1], np.finfo(float).eps)
    r2 = np.maximum(distances[:, 2], r1)
    logs = np.log(r2 / r1)
    valid = np.isfinite(logs) & (logs > 0)
    return float(valid.sum() / logs[valid].sum()) if valid.any() else np.nan


def mle_local(values: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(int(k), len(values) - 1)
    if k_eff < 3:
        return np.full(len(values), np.nan)
    distances = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto", n_jobs=-1).fit(values).kneighbors(values)[0]
    radii = np.maximum(distances[:, 1:], np.finfo(float).eps)
    outer = radii[:, -1:]
    denominator = np.sum(np.log(outer / radii[:, :-1]), axis=1)
    result = np.divide(
        k_eff - 2,
        denominator,
        out=np.full(len(values), np.nan),
        where=denominator > 0,
    )
    return result


def fisher_separability(values: np.ndarray, conditional_number: int = 10) -> tuple[float, str]:
    try:
        import skdim.id
    except ImportError:
        return np.nan, "not installed; run: pip install scikit-dimension"
    try:
        estimator = skdim.id.FisherS(conditional_number=conditional_number)
        estimator.fit(values)
        return float(estimator.dimension_), "ok"
    except Exception as error:
        return np.nan, f"failed: {type(error).__name__}: {error}"


def evaluate_id(
    values: np.ndarray,
    config: dict[str, Any],
    seeds: list[int],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    sample_size = min(int(config.get("sample_size", 100000)), len(values))
    if sample_size < 3:
        raise ValueError("Intrinsic-dimension evaluation requires at least 3 samples")
    k_global = min(int(config.get("k_global", 20)), sample_size - 1)
    k_local = min(int(config.get("k_local", 100)), sample_size - 1)
    rows = []
    first_local = np.full(len(values), np.nan)
    effective_seeds = (
        seeds
        if sample_size < len(values) or config.get("repeat_full_sample", False)
        else seeds[:1]
    )
    for repeat, seed in enumerate(effective_seeds):
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(values), size=sample_size, replace=False))
        subset = values[selected]
        local = mle_local(subset, k_local)
        if repeat == 0:
            first_local[selected] = local
        fisher, fisher_status = fisher_separability(
            subset, int(config.get("fisher_conditional_number", 10))
        )
        finite_local = local[np.isfinite(local)]
        rows.append({
            "seed": seed,
            "resampled": sample_size < len(values),
            "n_samples": sample_size,
            "ambient_dimension": values.shape[1],
            "twonn_id": twonn(subset),
            "mle_global_id": float(np.nanmean(mle_local(subset, k_global))),
            "participation_ratio_id": participation_ratio(subset),
            "fishers_id": fisher,
            "fishers_status": fisher_status,
            "local_mle_k": k_local,
            "local_id_mean": float(np.mean(finite_local)) if len(finite_local) else np.nan,
            "local_id_std": float(np.std(finite_local)) if len(finite_local) else np.nan,
            "local_id_p05": float(np.quantile(finite_local, 0.05)) if len(finite_local) else np.nan,
            "local_id_p50": float(np.quantile(finite_local, 0.50)) if len(finite_local) else np.nan,
            "local_id_p95": float(np.quantile(finite_local, 0.95)) if len(finite_local) else np.nan,
        })
    return rows, first_local
