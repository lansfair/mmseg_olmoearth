"""K-means, DBSCAN, and cosine retrieval for generic PT bundles."""

from __future__ import annotations

import hashlib
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .bundle import (
    BUNDLE_INPUT_FORMAT,
    FIXED_SEED,
    BundleSplit,
    bundle_paths,
    load_bundle_split,
    strict_json_dumps,
    write_json_atomic,
)
from .clustering import evaluate_dbscan, evaluate_kmeans
from .retrieval import evaluate_semantic_retrieval


TASK_PROTOCOLS = {
    "kmeans": "geofm_kmeans_v1",
    "dbscan": "geofm_dbscan_cosine_v1",
    "cosine_retrieval": "geofm_cosine_retrieval_v1",
}


@dataclass(frozen=True)
class SampledFeatures:
    split: BundleSplit
    indices: np.ndarray
    labels: np.ndarray
    features: np.ndarray
    class_counts: dict[int, int]
    unit: str


def label_sha256(labels: torch.Tensor, chunk_rows: int = 64) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(labels), chunk_rows):
        chunk = labels[start : start + chunk_rows].contiguous().numpy()
        digest.update(memoryview(chunk))
    return digest.hexdigest()


def balanced_indices(
    labels: torch.Tensor,
    *,
    per_class: int,
    ignore_label: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    if labels.ndim not in {1, 3}:
        raise ValueError(f"Labels must be [N] or [N,H,W], got {labels.shape}")
    if per_class < 1:
        raise ValueError("per_class must be positive")
    flat = labels.reshape(-1)
    valid = (flat != ignore_label) & (flat >= 0)
    classes = torch.unique(flat[valid]).sort().values.tolist()
    if not classes:
        raise ValueError("Split has no non-ignored class labels")
    rng = np.random.default_rng(FIXED_SEED)
    selected: list[np.ndarray] = []
    selected_labels: list[np.ndarray] = []
    counts: dict[int, int] = {}
    for raw_class in classes:
        class_id = int(raw_class)
        positions = torch.nonzero(flat == class_id, as_tuple=False).flatten().numpy()
        count = min(per_class, len(positions))
        counts[class_id] = count
        chosen = np.asarray(
            rng.choice(positions, size=count, replace=False), dtype=np.int64
        )
        selected.append(chosen)
        selected_labels.append(np.full(count, class_id, dtype=np.int64))
    indices = np.concatenate(selected)
    sampled_labels = np.concatenate(selected_labels)
    order = np.argsort(indices)
    return indices[order], sampled_labels[order], counts


def gather_features(
    split: BundleSplit,
    indices: np.ndarray,
    *,
    chunk_size: int = 8192,
) -> tuple[np.ndarray, str]:
    """Gather vector, pooled-scene, or dense-pixel features."""

    chunks: list[np.ndarray] = []
    if split.embeddings.ndim == 2:
        unit = "sample"
        for start in range(0, len(indices), chunk_size):
            selected = torch.from_numpy(indices[start : start + chunk_size])
            chunks.append(split.embeddings[selected].float().numpy())
    elif split.labels.ndim == 1:
        unit = "sample_from_spatial_mean"
        for start in range(0, len(indices), chunk_size):
            selected = torch.from_numpy(indices[start : start + chunk_size])
            values = split.embeddings[selected].float().mean(dim=(1, 2))
            chunks.append(values.numpy())
    else:
        unit = "pixel"
        label_height, label_width = map(int, split.labels.shape[1:])
        feature_height, feature_width = map(int, split.embeddings.shape[1:3])
        pixels_per_image = label_height * label_width
        samples = indices // pixels_per_image
        within = indices % pixels_per_image
        pixel_y, pixel_x = within // label_width, within % label_width
        feature_y = np.minimum(
            ((pixel_y + 0.5) * feature_height / label_height).astype(np.int64),
            feature_height - 1,
        )
        feature_x = np.minimum(
            ((pixel_x + 0.5) * feature_width / label_width).astype(np.int64),
            feature_width - 1,
        )
        for start in range(0, len(indices), chunk_size):
            end = min(start + chunk_size, len(indices))
            values = split.embeddings[
                torch.from_numpy(samples[start:end]),
                torch.from_numpy(feature_y[start:end]),
                torch.from_numpy(feature_x[start:end]),
            ]
            chunks.append(values.float().numpy())
    features = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if len(features) != len(indices):
        raise RuntimeError("Gathered feature count does not match sampled labels")
    if not np.isfinite(features).all():
        raise FloatingPointError("Sampled features contain NaN or Inf")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise FloatingPointError("Sampled features contain zero-norm vectors")
    return features / norms, unit


def sample_split(
    *,
    root: str | Path,
    dataset: str,
    model: str,
    split_name: str,
    per_class: int,
    ignore_label: int,
) -> SampledFeatures:
    path = bundle_paths(
        root, dataset=dataset, model=model, splits=(split_name,)
    )[split_name]
    split = load_bundle_split(path)
    indices, labels, counts = balanced_indices(
        split.labels, per_class=per_class, ignore_label=ignore_label
    )
    features, unit = gather_features(split, indices)
    return SampledFeatures(split, indices, labels, features, counts, unit)


def _runtime() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }


def _sample_audit(sampled: SampledFeatures, split_name: str) -> dict[str, Any]:
    return {
        "split": split_name,
        "bundle": sampled.split.audit,
        "label_sha256": label_sha256(sampled.split.labels),
        "sampling": "class_balanced_without_replacement",
        "representation_unit": sampled.unit,
        "feature_mapping": (
            "label_pixel_center_to_nearest_native_feature_token"
            if sampled.unit == "pixel"
            else sampled.unit
        ),
        "sampled_items": int(len(sampled.labels)),
        "sampled_class_counts": sampled.class_counts,
        "feature_dimension": int(sampled.features.shape[1]),
        "l2_normalized": True,
    }


def _base_report(
    *,
    task: str,
    dataset: str,
    model: str,
    input_audit: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    return {
        "task": task,
        "protocol": TASK_PROTOCOLS[task],
        "input_format": BUNDLE_INPUT_FORMAT,
        "dataset": dataset,
        "model": model,
        "seed": FIXED_SEED,
        "input_audit": input_audit,
        "runtime": _runtime(),
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_kmeans(
    *,
    root: str | Path,
    dataset: str,
    model: str,
    output_dir: str | Path,
    split_name: str = "test",
    per_class: int = 256,
    ignore_label: int = -1,
    clusters: int | None = None,
    n_init: int = 20,
    max_iter: int = 300,
) -> Path:
    started = time.perf_counter()
    if n_init < 1 or max_iter < 1:
        raise ValueError("n_init and max_iter must be positive")
    sampled = sample_split(
        root=root,
        dataset=dataset,
        model=model,
        split_name=split_name,
        per_class=per_class,
        ignore_label=ignore_label,
    )
    observed_classes = len(np.unique(sampled.labels))
    clusters_was_default = clusters is None
    clusters = observed_classes if clusters_was_default else int(clusters)
    if not 2 <= clusters <= len(sampled.features):
        raise ValueError("clusters must be between 2 and sampled item count")
    results = evaluate_kmeans(
        sampled.features,
        sampled.labels,
        {
            "cluster_counts": [clusters],
            "n_init": n_init,
            "max_iter": max_iter,
            "silhouette_sample_size": min(10000, len(sampled.features)),
        },
        FIXED_SEED,
    )
    report = _base_report(
        task="kmeans",
        dataset=dataset,
        model=model,
        input_audit={split_name: _sample_audit(sampled, split_name)},
        started=started,
    )
    report["config"] = {
        "split": split_name,
        "per_class": per_class,
        "ignore_label": ignore_label,
        "clusters": clusters,
        "clusters_defaulted_to_observed_classes": clusters_was_default,
        "n_init": n_init,
        "max_iter": max_iter,
        "distance_space": "L2-normalized embedding",
    }
    report["results"] = results
    return write_json_atomic(Path(output_dir) / "report.json", report)


def run_dbscan(
    *,
    root: str | Path,
    dataset: str,
    model: str,
    output_dir: str | Path,
    split_name: str = "test",
    per_class: int = 256,
    ignore_label: int = -1,
    min_samples: Iterable[int] = (5, 10, 20),
    eps_multipliers: Iterable[float] = (0.9, 1.0, 1.1),
) -> Path:
    started = time.perf_counter()
    min_samples = tuple(int(value) for value in min_samples)
    eps_multipliers = tuple(float(value) for value in eps_multipliers)
    if not min_samples or min(min_samples) < 2:
        raise ValueError("min_samples must contain values >= 2")
    if not eps_multipliers or min(eps_multipliers) <= 0:
        raise ValueError("eps_multipliers must contain positive values")
    sampled = sample_split(
        root=root,
        dataset=dataset,
        model=model,
        split_name=split_name,
        per_class=per_class,
        ignore_label=ignore_label,
    )
    if max(min_samples) > len(sampled.features):
        raise ValueError("min_samples cannot exceed sampled item count")
    results = evaluate_dbscan(
        sampled.features,
        sampled.labels,
        {
            "min_samples": list(min_samples),
            "eps_multipliers": list(eps_multipliers),
            "silhouette_sample_size": min(10000, len(sampled.features)),
        },
        FIXED_SEED,
    )
    report = _base_report(
        task="dbscan",
        dataset=dataset,
        model=model,
        input_audit={split_name: _sample_audit(sampled, split_name)},
        started=started,
    )
    report["config"] = {
        "split": split_name,
        "per_class": per_class,
        "ignore_label": ignore_label,
        "min_samples": list(min_samples),
        "eps_multipliers": list(eps_multipliers),
        "metric": "cosine",
        "eps_selection": "k-distance knee with 0.9-quantile fallback",
    }
    report["results"] = results
    return write_json_atomic(Path(output_dir) / "report.json", report)


def run_cosine_retrieval(
    *,
    root: str | Path,
    dataset: str,
    model: str,
    output_dir: str | Path,
    gallery_split: str = "train",
    query_split: str = "test",
    gallery_per_class: int = 512,
    query_per_class: int = 256,
    ignore_label: int = -1,
    k_values: Iterable[int] = (1, 5, 10, 20),
    batch_size: int = 512,
) -> Path:
    started = time.perf_counter()
    k_values = tuple(sorted({int(value) for value in k_values}))
    if not k_values or min(k_values) < 1:
        raise ValueError("k_values must contain positive integers")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if gallery_split == query_split:
        raise ValueError("gallery_split and query_split must differ")
    gallery = sample_split(
        root=root,
        dataset=dataset,
        model=model,
        split_name=gallery_split,
        per_class=gallery_per_class,
        ignore_label=ignore_label,
    )
    query = sample_split(
        root=root,
        dataset=dataset,
        model=model,
        split_name=query_split,
        per_class=query_per_class,
        ignore_label=ignore_label,
    )
    missing_query_classes = set(map(int, np.unique(query.labels))).difference(
        map(int, np.unique(gallery.labels))
    )
    if missing_query_classes:
        raise ValueError(
            "Query classes are absent from the gallery: "
            f"{sorted(missing_query_classes)}"
        )
    if gallery.unit != query.unit:
        raise ValueError("Gallery and query representation units differ")
    if gallery.features.shape[1] != query.features.shape[1]:
        raise ValueError("Gallery and query embedding dimensions differ")
    values = np.concatenate((gallery.features, query.features), axis=0)
    labels = np.concatenate((gallery.labels, query.labels), axis=0)
    sample_ids = np.asarray(
        [f"{gallery_split}:{value}" for value in gallery.indices]
        + [f"{query_split}:{value}" for value in query.indices]
    )
    split = np.asarray(
        [gallery_split] * len(gallery.features)
        + [query_split] * len(query.features)
    )
    summary, details = evaluate_semantic_retrieval(
        values,
        labels,
        sample_ids,
        split,
        {
            "gallery_split": gallery_split,
            "query_split": query_split,
            "k_values": list(k_values),
            "batch_size": batch_size,
        },
    )
    output_dir = Path(output_dir).resolve()
    details_path = output_dir / "details.jsonl"
    details_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = details_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in details:
            stream.write(strict_json_dumps(row, indent=None) + "\n")
    temporary.replace(details_path)
    report = _base_report(
        task="cosine_retrieval",
        dataset=dataset,
        model=model,
        input_audit={
            gallery_split: _sample_audit(gallery, gallery_split),
            query_split: _sample_audit(query, query_split),
        },
        started=started,
    )
    report["config"] = {
        "gallery_split": gallery_split,
        "query_split": query_split,
        "gallery_per_class": gallery_per_class,
        "query_per_class": query_per_class,
        "ignore_label": ignore_label,
        "k_values": list(k_values),
        "batch_size": batch_size,
        "metric": "cosine",
    }
    report["summary"] = summary[0]
    report["details"] = {
        "path": str(details_path),
        "rows": len(details),
        "format": "jsonl",
    }
    return write_json_atomic(output_dir / "report.json", report)
