from __future__ import annotations

import json
from typing import Any

import numpy as np


def evaluate_semantic_retrieval(
    values: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    split: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gallery_split = str(config.get("gallery_split", "train"))
    query_split = str(config.get("query_split", "test"))
    gallery = np.flatnonzero(split == gallery_split)
    queries = np.flatnonzero(split == query_split)
    if not len(gallery) or not len(queries):
        raise ValueError(
            f"Retrieval requires non-empty gallery={gallery_split} and query={query_split}"
        )
    same_collection = gallery_split == query_split
    gallery_capacity = len(gallery) - 1 if same_collection else len(gallery)
    if gallery_capacity < 1:
        raise ValueError("Retrieval gallery has no candidate after self-exclusion")
    requested_k = sorted({int(k) for k in config.get("k_values", [1, 5, 10])})
    if not requested_k or requested_k[0] < 1:
        raise ValueError("Retrieval k_values must contain positive integers")
    effective_k = [min(k, gallery_capacity) for k in requested_k]
    maximum_k = max(effective_k)
    batch_size = int(config.get("batch_size", 512))
    gallery_x = values[gallery]
    gallery_y = labels[gallery]
    gallery_ids = sample_ids[gallery]
    details: list[dict[str, Any]] = []
    accumulators = {
        k: {"hit": [], "precision": [], "recall": [], "ap": []}
        for k in requested_k
    }
    reciprocal_ranks = []

    for start in range(0, len(queries), batch_size):
        query_indices = queries[start:start + batch_size]
        similarities = values[query_indices] @ gallery_x.T
        if same_collection:
            similarities[sample_ids[query_indices, None] == gallery_ids[None, :]] = -np.inf
        order = np.argpartition(-similarities, maximum_k - 1, axis=1)[:, :maximum_k]
        selected_scores = np.take_along_axis(similarities, order, axis=1)
        rank_order = np.argsort(-selected_scores, axis=1)
        order = np.take_along_axis(order, rank_order, axis=1)
        selected_scores = np.take_along_axis(selected_scores, rank_order, axis=1)

        for row_index, query_index in enumerate(query_indices):
            ranked = order[row_index]
            ranked_labels = gallery_y[ranked]
            relevant = ranked_labels == labels[query_index]
            positive_mask = gallery_y == labels[query_index]
            if same_collection:
                positive_mask &= gallery_ids != sample_ids[query_index]
            total_relevant = int(positive_mask.sum())
            if total_relevant:
                best_positive_score = float(np.max(similarities[row_index, positive_mask]))
                first_rank = int(1 + np.sum(similarities[row_index] > best_positive_score))
            else:
                first_rank = None
            reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
            detail: dict[str, Any] = {
                "query_id": sample_ids[query_index],
                "query_label": labels[query_index],
                "query_split": query_split,
                "gallery_split": gallery_split,
                "positive_gallery_count": total_relevant,
                "first_positive_rank": first_rank,
                "top_ids": json.dumps(gallery_ids[ranked].tolist(), ensure_ascii=False),
                "top_labels": json.dumps(ranked_labels.tolist(), ensure_ascii=False),
                "top_similarities": json.dumps(
                    selected_scores[row_index].astype(float).tolist()
                ),
            }
            cumulative = np.cumsum(relevant)
            ranks = np.arange(1, len(relevant) + 1)
            precision_at_rank = cumulative / ranks
            for requested, effective in zip(requested_k, effective_k):
                hits = int(cumulative[effective - 1])
                hit = float(hits > 0)
                precision = hits / effective
                recall = hits / total_relevant if total_relevant else np.nan
                denominator = min(total_relevant, effective)
                ap = (
                    float(np.sum(precision_at_rank[:effective] * relevant[:effective]) / denominator)
                    if denominator
                    else np.nan
                )
                accumulators[requested]["hit"].append(hit)
                accumulators[requested]["precision"].append(precision)
                accumulators[requested]["recall"].append(recall)
                accumulators[requested]["ap"].append(ap)
                detail[f"hit_at_{requested}"] = hit
                detail[f"precision_at_{requested}"] = precision
                detail[f"recall_at_{requested}"] = recall
                detail[f"ap_at_{requested}"] = ap
            details.append(detail)

    summary: dict[str, Any] = {
        "algorithm": "cosine_semantic_retrieval",
        "gallery_split": gallery_split,
        "query_split": query_split,
        "gallery_samples": len(gallery),
        "query_samples": len(queries),
        "mrr": float(np.mean(reciprocal_ranks)),
    }
    for k in requested_k:
        summary[f"hit_rate_at_{k}"] = float(np.nanmean(accumulators[k]["hit"]))
        summary[f"precision_at_{k}"] = float(np.nanmean(accumulators[k]["precision"]))
        summary[f"recall_at_{k}"] = float(np.nanmean(accumulators[k]["recall"]))
        summary[f"map_at_{k}"] = float(np.nanmean(accumulators[k]["ap"]))
    return [summary], details
