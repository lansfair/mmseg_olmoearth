"""Compare original OLMoEarth eval embeddings with MMSeg exported embeddings.

The original eval code is expected to save a ``.pt`` file with keys such as
``val_embeddings`` and ``val_labels``. The MMSeg extraction script writes a
GeoTIFF manifest where each embedding is saved as CHW ``embedding.tif``.
This script converts the MMSeg tensors to NHWC and compares them sample by
sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_tif(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        return src.read()


def _to_numpy(tensor: Any) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _get_split_tensor(payload: dict[str, Any], split: str, kind: str) -> np.ndarray:
    candidates = [
        f"{split}_{kind}",
        f"{split}_embeddings" if kind == "embeddings" else f"{split}_labels",
    ]
    if split == "val":
        candidates.extend(
            [
                f"valid_{kind}",
                "embeddings" if kind == "embeddings" else "labels",
            ]
        )
    for key in candidates:
        value = payload.get(key)
        if value is not None:
            return _to_numpy(value)
    raise KeyError(
        f"Could not find {kind} for split '{split}' in .pt file. "
        f"Tried keys: {candidates}. Available keys: {sorted(payload.keys())}"
    )


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 4:
        raise ValueError(
            "Expected original embeddings as NHWC with shape [N,H,W,C], "
            f"got {embeddings.shape}"
        )
    return embeddings.astype(np.float32, copy=False)


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 3:
        raise ValueError(
            "Expected original labels as [N,H,W] or [N,1,H,W], "
            f"got {labels.shape}"
        )
    return labels


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64, copy=False)
    b_flat = b.reshape(-1).astype(np.float64, copy=False)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0:
        return float("nan")
    return float(np.dot(a_flat, b_flat) / denom)


def _compare_one(
    original_embedding: np.ndarray,
    original_label: np.ndarray,
    mmseg_embedding_path: Path,
    mmseg_label_path: Path,
) -> dict[str, Any]:
    mmseg_embedding_chw = _read_tif(mmseg_embedding_path)
    mmseg_embedding = np.moveaxis(mmseg_embedding_chw, 0, -1).astype(
        np.float32, copy=False
    )
    mmseg_label = _read_tif(mmseg_label_path).squeeze(0)

    if original_embedding.shape != mmseg_embedding.shape:
        raise ValueError(
            "Embedding shape mismatch: "
            f"original {original_embedding.shape}, mmseg {mmseg_embedding.shape} "
            f"for {mmseg_embedding_path}"
        )
    if original_label.shape != mmseg_label.shape:
        raise ValueError(
            "Label shape mismatch: "
            f"original {original_label.shape}, mmseg {mmseg_label.shape} "
            f"for {mmseg_label_path}"
        )

    diff = np.abs(original_embedding - mmseg_embedding)
    label_equal = bool(np.array_equal(original_label, mmseg_label))
    label_diff_pixels = int(np.count_nonzero(original_label != mmseg_label))
    return {
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "p95_abs_diff": float(np.percentile(diff, 95)) if diff.size else 0.0,
        "cosine": _cosine_similarity(original_embedding, mmseg_embedding),
        "label_equal": label_equal,
        "label_diff_pixels": label_diff_pixels,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare original OLMoEarth .pt embeddings with MMSeg GeoTIFF "
            "embeddings exported by projects/olmoearth/tools/extract_embeddings.py."
        )
    )
    parser.add_argument(
        "--original-pt",
        required=True,
        help="Path to the .pt file saved from olmoearth_pretrain.",
    )
    parser.add_argument(
        "--mmseg-root",
        required=True,
        help="MMSeg embedding output root that contains val.json/test.json.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "valid", "test", "train"],
        help="Split to compare. Use val for original valid split.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Number of samples to compare. Use 0 to compare all saved samples.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save detailed comparison results as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_pt = Path(args.original_pt)
    mmseg_root = Path(args.mmseg_root)
    split = "val" if args.split == "valid" else args.split
    manifest_path = mmseg_root / f"{split}.json"

    payload = torch.load(original_pt, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dict in {original_pt}, got {type(payload)}")

    original_embeddings = _normalize_embeddings(
        _get_split_tensor(payload, split, "embeddings")
    )
    original_labels = _normalize_labels(_get_split_tensor(payload, split, "labels"))

    manifest = _load_json(manifest_path)
    samples = sorted(
        manifest.get("samples", []), key=lambda sample: int(sample["source_index"])
    )
    limit = len(samples) if args.limit == 0 else min(args.limit, len(samples))
    limit = min(limit, original_embeddings.shape[0], original_labels.shape[0])

    if limit == 0:
        raise ValueError("No samples to compare.")

    results: list[dict[str, Any]] = []
    for i, sample in enumerate(samples[:limit]):
        embedding_path = mmseg_root / sample["embedding_path"]
        label_path = mmseg_root / sample["seg_map_path"]
        metrics = _compare_one(
            original_embeddings[i],
            original_labels[i],
            embedding_path,
            label_path,
        )
        row = {
            "index": i,
            "source_index": int(sample["source_index"]),
            "sample_id": sample.get("sample_id"),
            **metrics,
        }
        results.append(row)
        print(
            f"[{i:04d}] source_index={row['source_index']} "
            f"sample_id={row['sample_id']} "
            f"max={row['max_abs_diff']:.6g} "
            f"mean={row['mean_abs_diff']:.6g} "
            f"p95={row['p95_abs_diff']:.6g} "
            f"cos={row['cosine']:.9f} "
            f"label_equal={row['label_equal']} "
            f"label_diff_pixels={row['label_diff_pixels']}"
        )

    max_abs = max(row["max_abs_diff"] for row in results)
    mean_abs = float(np.mean([row["mean_abs_diff"] for row in results]))
    min_cos = min(row["cosine"] for row in results)
    all_labels_equal = all(row["label_equal"] for row in results)
    summary = {
        "original_pt": str(original_pt),
        "manifest": str(manifest_path),
        "split": split,
        "num_compared": limit,
        "original_embedding_shape": list(original_embeddings.shape),
        "original_label_shape": list(original_labels.shape),
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "min_cosine": min_cos,
        "all_labels_equal": all_labels_equal,
    }

    print("\nSummary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"summary": summary, "results": results},
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nWrote details to {output_path}")


if __name__ == "__main__":
    main()
