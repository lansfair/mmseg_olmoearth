#!/usr/bin/env python
"""Pack per-sample extraction manifests into one generic evaluation PT bundle."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack manifest embeddings and labels into a GeoFM PT bundle."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--labels-csv",
        type=Path,
        help="Optional sample-level CSV with sample_id,label columns.",
    )
    parser.add_argument("--id-column", default="sample_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="Create an inference bundle when the manifest has no labels.",
    )
    return parser.parse_args()


def _csv_labels(path: Path, id_column: str, label_column: str) -> dict[str, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if not rows.fieldnames or id_column not in rows.fieldnames:
            raise ValueError(f"CSV is missing ID column {id_column!r}")
        if label_column not in rows.fieldnames:
            raise ValueError(f"CSV is missing label column {label_column!r}")
        result: dict[str, int] = {}
        for row in rows:
            sample_id = str(row[id_column])
            if sample_id in result:
                raise ValueError(f"Duplicate CSV sample_id: {sample_id}")
            result[sample_id] = int(row[label_column])
    return result


def _load_raster(path: Path) -> torch.Tensor:
    try:
        import rasterio
    except ImportError as error:
        raise ImportError(f"Reading {path} requires rasterio") from error
    with rasterio.open(path) as source:
        return torch.from_numpy(source.read())


def _load_tensor(path: Path) -> torch.Tensor:
    if path.suffix.lower() in {".tif", ".tiff"}:
        return _load_raster(path)
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path}: expected one tensor")
    return value


def _manifest_member(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Manifest path escapes its root: {relative!r}") from error
    return path


def _embedding_to_bundle_layout(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().float().cpu()
    if value.ndim == 1:
        return value
    if value.ndim == 3:
        return value.permute(1, 2, 0).contiguous()
    raise ValueError(f"Per-sample embedding must be [D] or [D,H,W], got {value.shape}")


def _label_to_bundle_layout(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().cpu()
    if value.dtype.is_floating_point or value.dtype == torch.bool:
        raise TypeError(
            f"Per-sample label must contain integer class IDs, got {value.dtype}"
        )
    if value.ndim == 0:
        return value
    if value.ndim == 1 and value.numel() == 1:
        return value[0]
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"Per-sample label must be scalar or [H,W], got {value.shape}")
    return value


def _manifest_label_path(record: dict[str, object], sample_id: str) -> str:
    """Return the dense/scalar label path used by supported manifest producers."""

    label_path = record.get("label_path")
    seg_map_path = record.get("seg_map_path")
    if label_path is not None and seg_map_path is not None:
        if str(label_path) != str(seg_map_path):
            raise ValueError(
                f"{sample_id}: label_path and seg_map_path refer to different files"
            )
        return str(label_path)
    path = label_path if label_path is not None else seg_map_path
    if path is None:
        raise KeyError(
            f"{sample_id}: manifest has neither label_path nor seg_map_path; "
            "provide --labels-csv"
        )
    return str(path)


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    root = manifest_path.parent
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != "geofm_embedding_manifest_v1":
        raise ValueError("Unsupported manifest format")
    records = payload.get("samples")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest has no samples")
    csv_labels = (
        _csv_labels(args.labels_csv.resolve(), args.id_column, args.label_column)
        if args.labels_csv is not None
        else None
    )
    embeddings: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    per_sample_embedding_shape: tuple[int, ...] | None = None
    per_sample_label_shape: tuple[int, ...] | None = None
    label_dtype: torch.dtype | None = None
    expect_labels: bool | None = None
    sample_ids: list[str] = []
    source_shapes: list[tuple[int, int]] = []
    all_have_source_shape = True
    seen: set[str] = set()
    for index, record in enumerate(records):
        sample_id = str(record["sample_id"])
        if sample_id in seen:
            raise ValueError(f"Duplicate manifest sample_id: {sample_id}")
        seen.add(sample_id)
        sample_ids.append(sample_id)
        source_shape = record.get("source_shape")
        if source_shape is None:
            all_have_source_shape = False
        else:
            shape = tuple(int(value) for value in source_shape)
            if len(shape) != 2 or min(shape) < 1:
                raise ValueError(f"{sample_id}: invalid source_shape {source_shape}")
            source_shapes.append(shape)
        embedding_path = _manifest_member(root, str(record["embedding_path"]))
        embedding = _embedding_to_bundle_layout(_load_tensor(embedding_path))
        label: torch.Tensor | None
        if csv_labels is not None:
            if sample_id not in csv_labels:
                raise KeyError(f"CSV has no label for {sample_id}")
            label = torch.tensor(csv_labels[sample_id], dtype=torch.int64)
        elif "label_path" not in record and "seg_map_path" not in record:
            if not args.allow_unlabeled:
                raise KeyError(
                    f"{sample_id}: manifest has no label; pass --allow-unlabeled "
                    "for inference bundles"
                )
            label = None
        else:
            relative_label = _manifest_label_path(record, sample_id)
            label = _load_tensor(_manifest_member(root, str(relative_label)))
            label = _label_to_bundle_layout(label)

        embedding_shape = tuple(embedding.shape)
        current_has_label = label is not None
        if expect_labels is None:
            expect_labels = current_has_label
        elif current_has_label != expect_labels:
            raise ValueError("Manifest mixes labeled and unlabeled samples")
        if embeddings is None:
            per_sample_embedding_shape = embedding_shape
            embeddings = torch.empty(
                (len(records), *embedding_shape), dtype=embedding.dtype
            )
        else:
            if embedding_shape != per_sample_embedding_shape:
                raise ValueError(
                    f"{sample_id}: embedding shape {embedding_shape} differs from "
                    f"{per_sample_embedding_shape}"
                )
        if label is not None:
            current_label_shape = tuple(label.shape)
            if labels is None:
                per_sample_label_shape = current_label_shape
                label_dtype = label.dtype
                labels = torch.empty(
                    (len(records), *current_label_shape), dtype=label.dtype
                )
            if current_label_shape != per_sample_label_shape:
                raise ValueError(
                    f"{sample_id}: label shape {current_label_shape} differs from "
                    f"{per_sample_label_shape}"
                )
            if label.dtype != label_dtype:
                raise TypeError(
                    f"{sample_id}: label dtype {label.dtype} differs from {label_dtype}"
                )
        embeddings[index].copy_(embedding)
        if label is not None:
            assert labels is not None
            labels[index].copy_(label)

    if embeddings is None:
        raise RuntimeError("Manifest produced no tensors")
    if labels is not None and (embeddings.ndim, labels.ndim) not in {
        (2, 1),
        (4, 1),
        (4, 3),
    }:
        raise ValueError(
            "Unsupported embedding/label combination: bundle shapes "
            f"{embeddings.shape} and {labels.shape}"
        )
    bundle = {
        "embeddings": embeddings,
        "embedding_layout": "NHWD" if embeddings.ndim == 4 else "ND",
        "sample_ids": sample_ids,
    }
    if labels is not None:
        bundle["labels"] = labels
    if all_have_source_shape:
        bundle["source_shapes"] = torch.tensor(source_shapes, dtype=torch.int64)
    if not bool(torch.isfinite(bundle["embeddings"]).all()):
        raise FloatingPointError("Embeddings contain NaN or Inf")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(bundle, temporary)
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
