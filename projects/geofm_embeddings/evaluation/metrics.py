from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    silhouette_score,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    classes = np.unique(y_true)
    recalls = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    for label, value in zip(classes, recalls):
        safe_label = str(label).replace(" ", "_").replace(",", "_")
        result[f"recall__{safe_label}"] = float(value)
    return result


def hungarian_accuracy(y_true: np.ndarray, clusters: np.ndarray) -> float:
    true_codes, _ = _factorize(y_true)
    cluster_codes, _ = _factorize(clusters)
    matrix = confusion_matrix(true_codes, cluster_codes)
    row, col = linear_sum_assignment(matrix.max() - matrix)
    return float(matrix[row, col].sum() / len(y_true))


def purity_score(y_true: np.ndarray, clusters: np.ndarray) -> float:
    true_codes, _ = _factorize(y_true)
    cluster_codes, _ = _factorize(clusters)
    matrix = confusion_matrix(true_codes, cluster_codes)
    return float(matrix.max(axis=0).sum() / matrix.sum())


def cluster_metrics(
    values: np.ndarray,
    y_true: np.ndarray,
    clusters: np.ndarray,
    silhouette_sample_size: int,
    seed: int,
) -> dict[str, float]:
    unique_clusters = np.unique(clusters)
    silhouette = np.nan
    if 1 < len(unique_clusters) < len(clusters):
        sample_size = min(silhouette_sample_size, len(clusters))
        silhouette = silhouette_score(
            values,
            clusters,
            metric="cosine",
            sample_size=sample_size if sample_size < len(clusters) else None,
            random_state=seed,
        )
    return {
        "hungarian_accuracy": hungarian_accuracy(y_true, clusters),
        "nmi": normalized_mutual_info_score(y_true, clusters),
        "ari": adjusted_rand_score(y_true, clusters),
        "purity": purity_score(y_true, clusters),
        "silhouette_cosine": silhouette,
        "n_clusters": int(len(unique_clusters)),
    }


def _factorize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniques, codes = np.unique(np.asarray(values).astype(str), return_inverse=True)
    return codes, uniques
