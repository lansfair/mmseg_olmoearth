from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmbeddingTable:
    model: str
    sample_ids: np.ndarray
    values: np.ndarray


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_metadata(spec: dict[str, Any], base_dir: Path) -> pd.DataFrame:
    path = _resolve(base_dir, spec["path"])
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    id_col = spec.get("id_column", "sample_id")
    if id_col not in frame:
        raise ValueError(f"Metadata is missing id column: {id_col}")
    if frame[id_col].duplicated().any():
        raise ValueError("Metadata sample IDs must be unique")
    frame[id_col] = frame[id_col].astype(str)
    return frame


def _load_ids(path: Path, id_column: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        ids = np.load(path, allow_pickle=False)
    else:
        ids = pd.read_csv(path)[id_column].to_numpy()
    return ids.astype(str)


def geotiff_sample_id(path: Path, root: Path, mode: str) -> str:
    if mode == "parent":
        return path.parent.name
    if mode == "relative_parent":
        return path.parent.relative_to(root).as_posix()
    if mode == "relative_file":
        return path.relative_to(root).with_suffix("").as_posix()
    raise ValueError(f"Unsupported GeoTIFF id_mode: {mode}")


def pool_chw(values: np.ndarray, pooling: str) -> np.ndarray:
    if values.ndim != 3:
        raise ValueError(f"Expected GeoTIFF embedding [D,H,W], got {values.shape}")
    if pooling == "mean":
        return values.mean(axis=(1, 2))
    if pooling == "mean_max":
        return np.concatenate([values.mean(axis=(1, 2)), values.max(axis=(1, 2))])
    if pooling == "stats":
        return np.concatenate([
            values.mean(axis=(1, 2)),
            values.std(axis=(1, 2)),
            values.min(axis=(1, 2)),
            values.max(axis=(1, 2)),
        ])
    raise ValueError(f"Unsupported GeoTIFF pooling: {pooling}")


def _read_exported_embedding(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError as error:
            raise ImportError("GeoTIFF input requires rasterio") from error
        with rasterio.open(path) as source:
            return source.read().astype(np.float32, copy=False)
    if suffix == ".pt":
        try:
            import torch
        except ImportError as error:
            raise ImportError("PT embedding input requires PyTorch") from error
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            for key in ("embedding", "embeddings", "features", "tensor"):
                if key in payload:
                    payload = payload[key]
                    break
            else:
                raise ValueError(f"No embedding tensor found in {path}")
        if hasattr(payload, "detach"):
            payload = payload.detach().float().cpu().numpy()
        return np.asarray(payload, dtype=np.float32)
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    raise ValueError(f"Unsupported exported embedding file: {path}")


def _vectorize_export(values: np.ndarray, pooling: str, path: Path) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[0] == 1:
        return values[0]
    if values.ndim == 3:
        return pool_chw(values, pooling)
    raise ValueError(
        f"Expected global [D] or dense [D,H,W] embedding at {path}, got {values.shape}"
    )


def _manifest_paths(root_or_manifest: Path, spec: dict[str, Any]) -> list[Path]:
    if root_or_manifest.is_file():
        return [root_or_manifest]
    configured = spec.get("manifest_files")
    if configured:
        paths = [root_or_manifest / str(value) for value in configured]
    else:
        splits = spec.get("splits", ["train", "val", "test"])
        paths = [root_or_manifest / f"{split}.json" for split in splits]
    paths = [path for path in paths if path.exists()]
    if not paths:
        raise FileNotFoundError(f"No GeoFM manifest files found under {root_or_manifest}")
    return paths


def _load_geofm_manifests(
    spec: dict[str, Any], root_or_manifest: Path
) -> tuple[np.ndarray, np.ndarray]:
    pooling = str(spec.get("pooling", "mean"))
    id_mode = str(spec.get("id_mode", "sample_id"))
    if id_mode not in {"sample_id", "split_sample_id"}:
        raise ValueError("GeoFM manifest id_mode must be sample_id or split_sample_id")
    vectors: list[np.ndarray] = []
    ids: list[str] = []
    expected_dim: int | None = None
    for manifest_path in _manifest_paths(root_or_manifest, spec):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "geofm_embedding_manifest_v1":
            raise ValueError(f"Unsupported GeoFM manifest format: {manifest_path}")
        split_name = str(manifest.get("split", manifest_path.stem))
        for record in manifest.get("samples", []):
            sample_id = str(record["sample_id"])
            if id_mode == "split_sample_id":
                sample_id = f"{split_name}/{sample_id}"
            embedding_path = Path(record["embedding_path"])
            if not embedding_path.is_absolute():
                embedding_path = manifest_path.parent / embedding_path
            vector = _vectorize_export(
                _read_exported_embedding(embedding_path), pooling, embedding_path
            )
            if expected_dim is None:
                expected_dim = int(vector.shape[0])
            elif vector.shape != (expected_dim,):
                raise ValueError(
                    f"Pooled embedding dimension differs: expected {expected_dim}, "
                    f"got {vector.shape} at {embedding_path}"
                )
            vectors.append(vector)
            ids.append(sample_id)
    if not vectors:
        raise ValueError("GeoFM manifests contain no embedding records")
    return np.stack(vectors), np.asarray(ids)


def _load_geotiff_folder(spec: dict[str, Any], root: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import rasterio
    except ImportError as error:
        raise ImportError("GeoTIFF input requires rasterio") from error
    filename = spec.get("embedding_filename", "embedding.tif")
    paths = sorted(root.rglob(filename))
    if not paths:
        raise FileNotFoundError(f"No {filename} files found under {root}")
    pooling = spec.get("pooling", "mean")
    id_mode = spec.get("id_mode", "relative_parent")
    vectors = []
    ids = []
    expected_dimension = None
    for path in paths:
        with rasterio.open(path) as source:
            array = source.read().astype(np.float32, copy=False)
        vector = pool_chw(array, pooling)
        if expected_dimension is None:
            expected_dimension = vector.shape
        elif vector.shape != expected_dimension:
            raise ValueError(
                f"Pooled GeoTIFF dimensions differ: expected {expected_dimension}, "
                f"got {vector.shape} at {path}"
            )
        vectors.append(vector)
        ids.append(geotiff_sample_id(path, root, id_mode))
    return np.stack(vectors), np.asarray(ids)


def load_embedding(spec: dict[str, Any], base_dir: Path) -> EmbeddingTable:
    path = _resolve(base_dir, spec["path"])
    fmt = spec.get("format", path.suffix.lstrip(".")).lower()
    model = spec["name"]
    id_key = spec.get("id_key", "sample_id")
    embedding_key = spec.get("embedding_key", "embeddings")

    if fmt in {"geofm_manifest", "geofm_manifests"}:
        values, ids = _load_geofm_manifests(spec, path)
    elif fmt in {"geotiff", "tif_folder"}:
        values, ids = _load_geotiff_folder(spec, path)
    elif fmt == "npz":
        with np.load(path, allow_pickle=False) as archive:
            values = archive[embedding_key]
            ids = archive[id_key]
    elif fmt == "npy":
        values = np.load(path, mmap_mode=spec.get("mmap_mode"))
        ids_path = _resolve(base_dir, spec["ids_path"])
        ids = _load_ids(ids_path, spec.get("id_column", "sample_id"))
    elif fmt in {"csv", "parquet"}:
        frame = pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path)
        ids = frame[id_key].to_numpy()
        feature_cols = spec.get("feature_columns")
        if feature_cols is None:
            prefix = spec.get("feature_prefix", "embedding_")
            feature_cols = [c for c in frame.columns if c.startswith(prefix)]
        if not feature_cols:
            raise ValueError(f"No embedding columns found for {model}")
        values = frame[feature_cols].to_numpy()
    else:
        raise ValueError(f"Unsupported embedding format: {fmt}")

    values = np.asarray(values)
    ids = np.asarray(ids).astype(str)
    if values.ndim != 2:
        raise ValueError(f"{model}: expected [samples, dimensions], got {values.shape}")
    if len(ids) != len(values):
        raise ValueError(f"{model}: {len(ids)} IDs for {len(values)} embeddings")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{model}: sample IDs must be unique")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{model}: embeddings must be numeric")
    return EmbeddingTable(model, ids, values.astype(np.float32, copy=False))


def align_embedding(
    embedding: EmbeddingTable,
    metadata: pd.DataFrame,
    id_column: str,
    nonfinite_policy: str = "error",
) -> tuple[np.ndarray, pd.DataFrame]:
    if nonfinite_policy not in {"error", "drop"}:
        raise ValueError("nonfinite_policy must be 'error' or 'drop'")
    positions = pd.Series(np.arange(len(embedding.sample_ids)), index=embedding.sample_ids)
    keep = metadata[id_column].isin(positions.index)
    aligned_meta = metadata.loc[keep].copy().reset_index(drop=True)
    if aligned_meta.empty:
        raise ValueError(f"{embedding.model}: no sample IDs overlap with metadata")
    row_ids = positions.loc[aligned_meta[id_column]].to_numpy(dtype=int)
    values = np.asarray(embedding.values[row_ids], dtype=np.float32)
    finite = np.isfinite(values).all(axis=1)
    if not finite.all():
        count = int((~finite).sum())
        if nonfinite_policy == "error":
            raise ValueError(
                f"{embedding.model}: {count} aligned embedding rows contain NaN or Inf"
            )
        aligned_meta = aligned_meta.loc[finite].reset_index(drop=True)
        values = values[finite]
    return values, aligned_meta


def embedding_audit(
    embedding: EmbeddingTable,
    metadata: pd.DataFrame,
    id_column: str,
) -> dict[str, Any]:
    metadata_ids = set(metadata[id_column].astype(str))
    embedding_ids = set(embedding.sample_ids.astype(str))
    overlap = metadata_ids & embedding_ids
    finite_rows = np.isfinite(embedding.values).all(axis=1)
    norms = np.linalg.norm(embedding.values[finite_rows], axis=1)
    return {
        "model": embedding.model,
        "metadata_samples": len(metadata_ids),
        "embedding_samples": len(embedding_ids),
        "overlap_samples": len(overlap),
        "missing_from_embedding": len(metadata_ids - embedding_ids),
        "extra_in_embedding": len(embedding_ids - metadata_ids),
        "nonfinite_rows": int((~finite_rows).sum()),
        "embedding_dimension": int(embedding.values.shape[1]),
        "embedding_dtype": str(embedding.values.dtype),
        "zero_norm_rows": int(np.sum(norms == 0)),
        "norm_mean": float(norms.mean()) if len(norms) else np.nan,
        "norm_std": float(norms.std()) if len(norms) else np.nan,
        "norm_p05": float(np.quantile(norms, 0.05)) if len(norms) else np.nan,
        "norm_p95": float(np.quantile(norms, 0.95)) if len(norms) else np.nan,
    }


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")
