from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from .metrics import cluster_metrics


def evaluate_kmeans(
    values: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
    seeds: list[int],
) -> list[dict[str, Any]]:
    class_count = len(np.unique(labels))
    requested = config.get("cluster_counts", [class_count])
    counts = [class_count if value == "classes" else int(value) for value in requested]
    rows: list[dict[str, Any]] = []
    assignments: dict[int, list[np.ndarray]] = {count: [] for count in counts}
    for count in counts:
        for seed in seeds:
            model = KMeans(
                n_clusters=count,
                n_init=int(config.get("n_init", 20)),
                max_iter=int(config.get("max_iter", 300)),
                random_state=seed,
            )
            predicted = model.fit_predict(values)
            assignments[count].append(predicted)
            metrics = cluster_metrics(
                values,
                labels,
                predicted,
                int(config.get("silhouette_sample_size", 10000)),
                seed,
            )
            rows.append({"algorithm": "kmeans", "seed": seed, "parameter": count, **metrics})

        stability = []
        for i in range(len(assignments[count])):
            for j in range(i + 1, len(assignments[count])):
                stability.append(adjusted_rand_score(assignments[count][i], assignments[count][j]))
        value = float(np.mean(stability)) if stability else np.nan
        for row in rows:
            if row["algorithm"] == "kmeans" and row["parameter"] == count:
                row["stability_ari"] = value
    return rows
def estimate_dbscan_eps(values: np.ndarray, min_samples: int) -> float:
    neighbors = NearestNeighbors(n_neighbors=min_samples, metric="cosine", algorithm="brute")
    distances = neighbors.fit(values).kneighbors(return_distance=True)[0][:, -1]
    curve = np.sort(distances)
    if len(curve) < 3 or np.allclose(curve[0], curve[-1]):
        return float(np.quantile(curve, 0.9))
    x = np.linspace(0.0, 1.0, len(curve))
    y = (curve - curve[0]) / (curve[-1] - curve[0])
    knee = int(np.argmax(y - x))
    if knee < len(curve) // 2:
        return float(np.quantile(curve, 0.9))
    return float(curve[knee])


def evaluate_dbscan(
    values: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for min_samples in config.get("min_samples", [5, 10, 20]):
        base_eps = estimate_dbscan_eps(values, int(min_samples))
        for multiplier in config.get("eps_multipliers", [0.9, 1.0, 1.1]):
            eps = max(base_eps * float(multiplier), np.finfo(float).eps)
            predicted = DBSCAN(
                eps=eps,
                min_samples=int(min_samples),
                metric="cosine",
                algorithm="brute",
                n_jobs=-1,
            ).fit_predict(values)
            non_noise = predicted != -1
            row: dict[str, Any] = {
                "algorithm": "dbscan",
                "seed": seed,
                "parameter": int(min_samples),
                "eps": eps,
                "eps_multiplier": multiplier,
                "noise_ratio": float(1.0 - non_noise.mean()),
                "coverage": float(non_noise.mean()),
            }
            row.update(
                {f"all_{key}": value for key, value in cluster_metrics(
                    values,
                    labels,
                    predicted,
                    int(config.get("silhouette_sample_size", 10000)),
                    seed,
                ).items()}
            )
            if non_noise.sum() >= 3 and len(np.unique(predicted[non_noise])) >= 1:
                row.update(
                    {f"covered_{key}": value for key, value in cluster_metrics(
                        values[non_noise],
                        labels[non_noise],
                        predicted[non_noise],
                        int(config.get("silhouette_sample_size", 10000)),
                        seed,
                    ).items()}
                )
            rows.append(row)
    return rows
