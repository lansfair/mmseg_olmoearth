"""Paper-aligned cosine kNN evaluation for frozen classification embeddings."""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from .bundle import (
    BUNDLE_INPUT_FORMAT,
    FIXED_SEED,
    BundleSplit,
    bundle_paths,
    load_bundle_split,
    write_json_atomic,
)
from .metrics import classification_metrics


KNN_PROTOCOL = "olmoearth_cosine_knn20_softmax_vote_v1"


def _classification_rows(
    split: BundleSplit,
    *,
    ignore_label: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if split.labels.ndim != 1:
        raise ValueError(
            "kNN requires sample-level labels [N]; use linear for dense segmentation"
        )
    valid = (split.labels != ignore_label) & (split.labels >= 0)
    if not bool(valid.any()):
        raise ValueError(f"{split.path}: no non-ignored classification samples")
    labels = split.labels[valid].to(torch.int64)
    if split.embeddings.ndim == 2:
        embeddings = split.embeddings[valid]
        pooling = "none"
    else:
        embeddings = split.embeddings[valid].float().mean(dim=(1, 2))
        pooling = "spatial_mean"
    embeddings = embeddings.float()
    if not bool(torch.isfinite(embeddings).all()):
        raise FloatingPointError(f"{split.path}: embeddings contain NaN or Inf")
    norms = torch.linalg.vector_norm(embeddings, dim=1)
    if bool((norms <= torch.finfo(torch.float32).eps).any()):
        raise FloatingPointError(f"{split.path}: embeddings contain zero-norm rows")
    return F.normalize(embeddings, dim=1), labels, pooling


def _class_mapping(
    train_labels: torch.Tensor,
    query_labels: torch.Tensor,
) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor]:
    classes = tuple(int(value) for value in torch.unique(train_labels).sort().values)
    if len(classes) < 2:
        raise ValueError("kNN training labels must contain at least two classes")
    unknown = set(int(value) for value in torch.unique(query_labels)).difference(
        classes
    )
    if unknown:
        raise ValueError(
            f"Query labels contain classes absent from train: {sorted(unknown)}"
        )
    train_mapped = torch.empty_like(train_labels)
    query_mapped = torch.empty_like(query_labels)
    for contiguous, original in enumerate(classes):
        train_mapped[train_labels == original] = contiguous
        query_mapped[query_labels == original] = contiguous
    return classes, train_mapped, query_mapped


def cosine_knn_predict(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    *,
    num_classes: int,
    k: int = 20,
    temperature: float = 0.07,
    batch_size: int = 2000,
    device: str = "cuda",
) -> torch.Tensor:
    """Cosine kNN with the OLMoEarth paper's softmax-weighted vote."""

    if not 1 <= k <= len(train_embeddings):
        raise ValueError("k must be between 1 and the training sample count")
    if temperature <= 0 or batch_size < 1:
        raise ValueError("temperature and batch_size must be positive")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    gallery = train_embeddings.to(torch_device)
    gallery_labels = train_labels.to(torch_device)
    predictions: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(query_embeddings), batch_size):
            query = query_embeddings[start : start + batch_size].to(torch_device)
            similarities = query @ gallery.T
            top_similarities, top_indices = torch.topk(similarities, k=k, dim=1)
            top_labels = gallery_labels[top_indices]
            one_hot = F.one_hot(top_labels, num_classes=num_classes)
            weights = torch.exp(top_similarities / temperature).unsqueeze(-1)
            predictions.append((weights * one_hot).sum(dim=1).argmax(dim=1).cpu())
    return torch.cat(predictions)


def run_knn(
    *,
    root: str | Path,
    dataset: str,
    model: str,
    output_dir: str | Path,
    split_name: str = "test",
    k: int = 20,
    temperature: float = 0.07,
    batch_size: int = 2000,
    ignore_label: int = -1,
    device: str = "cuda",
) -> Path:
    if split_name not in {"valid", "test"}:
        raise ValueError("split_name must be 'valid' or 'test'")
    started = time.perf_counter()
    paths = bundle_paths(
        root, dataset=dataset, model=model, splits=("train", split_name)
    )
    train = load_bundle_split(paths["train"])
    query = load_bundle_split(paths[split_name])
    train_x, train_y, train_pooling = _classification_rows(
        train, ignore_label=ignore_label
    )
    query_x, query_y, query_pooling = _classification_rows(
        query, ignore_label=ignore_label
    )
    if train_x.shape[1] != query_x.shape[1]:
        raise ValueError("Train and query embedding dimensions differ")
    if train_pooling != query_pooling:
        raise ValueError("Train and query pooling modes differ")
    classes, train_mapped, query_mapped = _class_mapping(train_y, query_y)
    predicted = cosine_knn_predict(
        train_x,
        train_mapped,
        query_x,
        num_classes=len(classes),
        k=k,
        temperature=temperature,
        batch_size=batch_size,
        device=device,
    )
    true_numpy = query_mapped.numpy()
    predicted_numpy = predicted.numpy()
    report: dict[str, Any] = {
        "task": "knn",
        "protocol": KNN_PROTOCOL,
        "input_format": BUNDLE_INPUT_FORMAT,
        "dataset": dataset,
        "model": model,
        "split": split_name,
        "seed": FIXED_SEED,
        "config": {
            "k": k,
            "metric": "cosine",
            "vote": "softmax_weighted",
            "temperature": temperature,
            "batch_size": batch_size,
            "ignore_label": ignore_label,
            "device": device,
            "pooling": train_pooling,
            "gallery_split": "train",
            "query_split": split_name,
        },
        "class_values": list(classes),
        "input_audit": {"train": train.audit, split_name: query.audit},
        "metrics": classification_metrics(true_numpy, predicted_numpy),
        "confusion_matrix": confusion_matrix(
            true_numpy, predicted_numpy, labels=np.arange(len(classes))
        ).tolist(),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    return write_json_atomic(Path(output_dir) / "report.json", report)
