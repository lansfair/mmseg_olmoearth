from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    return obj


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(payload), f, indent=2)


def label_stats(
    labels: list[np.ndarray],
    ignore_index: int,
    num_classes: int,
) -> dict[str, Any]:
    if not labels:
        return {
            "num_samples": 0,
            "unique_values": [],
            "class_pixel_counts": [0 for _ in range(num_classes)],
            "ignore_pixel_count": 0,
            "out_of_range_values": [],
        }

    flat = np.concatenate([label.reshape(-1) for label in labels])
    unique = np.unique(flat)
    class_counts = [
        int(np.count_nonzero(flat == class_idx))
        for class_idx in range(num_classes)
    ]
    out_of_range = unique[
        ((unique < 0) | (unique >= num_classes)) & (unique != ignore_index)
    ]
    return {
        "num_samples": len(labels),
        "unique_values": unique.astype(np.int64).tolist(),
        "class_pixel_counts": class_counts,
        "ignore_pixel_count": int(np.count_nonzero(flat == ignore_index)),
        "out_of_range_values": out_of_range.astype(np.int64).tolist(),
    }


def validate_labels(
    label: np.ndarray,
    num_classes: int,
    ignore_index: int,
    sample_id: str,
) -> None:
    invalid = (
        ((label < 0) | (label >= num_classes))
        & (label != ignore_index)
    )
    if np.any(invalid):
        values = np.unique(label[invalid]).astype(np.int64).tolist()
        raise ValueError(
            f"{sample_id} has label values outside [0, {num_classes}) "
            f"and ignore_index={ignore_index}: {values}"
        )
