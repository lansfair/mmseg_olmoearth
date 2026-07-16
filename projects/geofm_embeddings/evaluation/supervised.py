from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import classification_metrics


def sample_per_class(
    indices: np.ndarray,
    labels: np.ndarray,
    budget: int | str,
    seed: int,
) -> np.ndarray:
    if budget == "all":
        return indices
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(labels[indices]):
        candidates = indices[labels[indices] == label]
        count = min(int(budget), len(candidates))
        selected.extend(rng.choice(candidates, size=count, replace=False))
    return np.asarray(sorted(selected), dtype=int)


def evaluate_knn(
    values: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
    config: dict[str, Any],
    seeds: list[int],
) -> list[dict[str, Any]]:
    train_pool = np.flatnonzero(split == "train")
    test = np.flatnonzero(split == "test")
    if not len(train_pool) or not len(test):
        raise ValueError("KNN evaluation requires non-empty train and test splits")
    requested_k = sorted({int(value) for value in config.get("k_values", [1, 3, 5, 7, 9])})
    if not requested_k or requested_k[0] < 1:
        raise ValueError("KNN k_values must contain positive integers")
    weights = config.get("weights", "uniform")
    if weights not in {"uniform", "distance"}:
        raise ValueError("KNN weights must be 'uniform' or 'distance'")
    rows = []
    for budget in config.get("budgets_per_class", [1, 5, 10, 20, 50, "all"]):
        budget_seeds = seeds if budget != "all" or config.get("repeat_all_seeds", False) else seeds[:1]
        for seed in budget_seeds:
            train = sample_per_class(train_pool, labels, budget, seed)
            if not len(train):
                raise ValueError(f"KNN budget {budget!r} selected no training samples")
            max_k = min(max(requested_k), len(train))
            neighbors = NearestNeighbors(
                n_neighbors=max_k,
                metric="cosine",
                algorithm="brute",
                n_jobs=int(config.get("n_jobs", -1)),
            ).fit(values[train])
            distances, neighbor_indices = neighbors.kneighbors(values[test], return_distance=True)
            classes, train_codes = np.unique(labels[train], return_inverse=True)
            neighbor_codes = train_codes[neighbor_indices]
            neighbor_labels = labels[train][neighbor_indices]
            for k in requested_k:
                effective_k = min(int(k), len(train))
                codes = neighbor_codes[:, :effective_k]
                if weights == "uniform":
                    vote_weights = np.ones_like(codes, dtype=np.float64)
                else:
                    selected_distances = distances[:, :effective_k]
                    zero = selected_distances == 0
                    rows_with_zero = zero.any(axis=1, keepdims=True)
                    vote_weights = np.divide(
                        1.0,
                        selected_distances,
                        out=np.zeros_like(selected_distances, dtype=np.float64),
                        where=selected_distances > 0,
                    )
                    vote_weights = np.where(rows_with_zero, zero.astype(np.float64), vote_weights)
                votes = np.zeros((len(test), len(classes)), dtype=np.float64)
                row_index = np.repeat(np.arange(len(test)), effective_k)
                np.add.at(votes, (row_index, codes.reshape(-1)), vote_weights.reshape(-1))
                predicted = classes[votes.argmax(axis=1)]
                neighborhood_purity = float(
                    np.mean(neighbor_labels[:, :effective_k] == labels[test, None])
                )
                rows.append({
                    "algorithm": "knn",
                    "seed": seed,
                    "budget_per_class": budget,
                    "k": int(k),
                    "effective_k": effective_k,
                    "n_train": len(train),
                    "n_test": len(test),
                    "weights": weights,
                    "neighborhood_purity": neighborhood_purity,
                    **classification_metrics(labels[test], predicted),
                })
    return rows

def evaluate_linear_probe(
    values: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
    config: dict[str, Any],
    seeds: list[int],
) -> list[dict[str, Any]]:
    train_pool = np.flatnonzero(split == "train")
    test = np.flatnonzero(split == "test")
    rows = []
    for budget in config.get("budgets_per_class", [1, 5, 10, 20, 50, "all"]):
        for seed in seeds:
            train = sample_per_class(train_pool, labels, budget, seed)
            min_class = min(np.sum(labels[train] == label) for label in np.unique(labels[train]))
            folds = min(int(config.get("cv_folds", 3)), int(min_class))
            if folds < 2:
                rows.append({
                    "algorithm": "linear_probe",
                    "seed": seed,
                    "budget_per_class": budget,
                    "status": "skipped: fewer than 2 samples in a class",
                })
                continue
            pipeline = Pipeline([
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(
                    max_iter=int(config.get("max_iter", 2000)),
                    class_weight=config.get("class_weight"),
                    random_state=seed,
                )),
            ])
            search = GridSearchCV(
                pipeline,
                {"classifier__C": config.get("c_values", [0.001, 0.01, 0.1, 1, 10, 100])},
                scoring="f1_macro",
                cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
                n_jobs=-1,
                refit=True,
            )
            search.fit(values[train], labels[train])
            predicted = search.predict(values[test])
            rows.append({
                "algorithm": "linear_probe",
                "seed": seed,
                "budget_per_class": budget,
                "n_train": len(train),
                "n_test": len(test),
                "best_c": search.best_params_["classifier__C"],
                "cv_macro_f1": search.best_score_,
                "status": "ok",
                **classification_metrics(labels[test], predicted),
            })
    return rows
