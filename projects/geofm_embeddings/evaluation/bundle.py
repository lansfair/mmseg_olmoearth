"""Dataset-independent loading and reporting for PT embedding bundles."""

from __future__ import annotations

import json
import math
import mmap
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor


BUNDLE_INPUT_FORMAT = "geofm_pt_bundle_v1"
FIXED_SEED = 42
_TORCH_MMAP_OPTIONS_LOCK = threading.Lock()


@dataclass(frozen=True)
class BundleSplit:
    path: Path
    embeddings: Tensor
    labels: Tensor
    embedding_layout: str

    @property
    def tensor_layout(self) -> str:
        """Describe storage only; evaluator selection remains external."""

        if self.embeddings.ndim == 4 and self.labels.ndim == 3:
            return "dense_grid_labels"
        if self.embeddings.ndim == 4 and self.labels.ndim == 1:
            return "spatial_sample_vectors"
        return "sample_vectors"

    @property
    def audit(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "size_bytes": self.path.stat().st_size,
            "tensor_layout": self.tensor_layout,
            "embedding_shape": list(self.embeddings.shape),
            "embedding_layout": self.embedding_layout,
            "embedding_dtype": str(self.embeddings.dtype),
            "label_shape": list(self.labels.shape),
            "label_dtype": str(self.labels.dtype),
        }


def _load_shared_mmap(path: Path) -> Any:
    """Load immutable tensor storage through a shared file mapping."""

    set_options = getattr(torch.serialization, "set_default_mmap_options", None)
    map_shared = getattr(mmap, "MAP_SHARED", None)
    if set_options is not None and map_shared is not None:
        with set_options(map_shared):
            return torch.load(path, map_location="cpu", weights_only=True, mmap=True)

    if map_shared is None:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)

    with _TORCH_MMAP_OPTIONS_LOCK:
        original_from_file = torch.UntypedStorage.from_file

        def from_file_shared(filename, _shared, size):
            return original_from_file(filename, True, size)

        torch.UntypedStorage.from_file = staticmethod(from_file_shared)
        try:
            return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        finally:
            torch.UntypedStorage.from_file = original_from_file


def load_bundle_split(path: str | Path) -> BundleSplit:
    """Load a dense or global embedding bundle and enforce its tensor contract."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = _load_shared_mmap(path)
    except TypeError as error:
        raise RuntimeError(
            "PT bundles require torch.load(..., mmap=True, weights_only=True)"
        ) from error
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: bundle must be a dictionary")
    required = {"embeddings", "labels"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"{path}: missing bundle keys {sorted(missing)}")
    embeddings, labels = payload["embeddings"], payload["labels"]
    if not isinstance(embeddings, Tensor) or not isinstance(labels, Tensor):
        raise TypeError(f"{path}: embeddings and labels must be torch tensors")
    if embeddings.ndim not in {2, 4}:
        raise ValueError(
            f"{path}: embeddings must be [N,D] or [N,H,W,D], got {embeddings.shape}"
        )
    layout = str(payload.get("embedding_layout", "NHWD" if embeddings.ndim == 4 else "ND"))
    expected_layouts = {2: {"ND"}, 4: {"NHWD", "NDHW"}}[embeddings.ndim]
    if layout not in expected_layouts:
        raise ValueError(
            f"{path}: embedding_layout must be one of {sorted(expected_layouts)}, "
            f"got {layout!r}"
        )
    if layout == "NDHW":
        embeddings = embeddings.permute(0, 2, 3, 1)
    if labels.ndim not in {1, 3}:
        raise ValueError(f"{path}: labels must be [N] or [N,H,W], got {labels.shape}")
    if embeddings.ndim == 2 and labels.ndim != 1:
        raise ValueError(f"{path}: [N,D] embeddings require [N] labels")
    if len(embeddings) != len(labels):
        raise ValueError(
            f"{path}: embedding/label rows differ: {len(embeddings)} != {len(labels)}"
        )
    if len(embeddings) == 0:
        raise ValueError(f"{path}: split is empty")
    if not embeddings.dtype.is_floating_point:
        raise TypeError(f"{path}: embeddings must be floating point")
    if labels.dtype.is_floating_point or labels.dtype == torch.bool:
        raise TypeError(f"{path}: labels must contain integer class IDs")
    if embeddings.ndim == 4 and min(map(int, embeddings.shape[1:3])) < 1:
        raise ValueError(f"{path}: feature grid cannot be empty")
    if labels.ndim == 3 and min(map(int, labels.shape[1:])) < 1:
        raise ValueError(f"{path}: label grid cannot be empty")
    return BundleSplit(
        path=path,
        embeddings=embeddings,
        labels=labels,
        embedding_layout="NHWD" if embeddings.ndim == 4 else "ND",
    )


def _safe_component(value: str, field: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{field} must be one directory name, got {value!r}")
    return value


def resolve_bundle_directory(
    root: str | Path,
    *,
    dataset: str,
    model: str,
) -> Path:
    """Resolve canonical ``dataset/model`` or released legacy ``model/dataset``."""

    root = Path(root).resolve()
    dataset = _safe_component(dataset, "dataset")
    model = _safe_component(model, "model")
    canonical = root / dataset / model
    legacy = root / model / dataset
    candidates = [path for path in (canonical, legacy) if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"No bundle directory found at {canonical} or {legacy}"
        )
    if len(candidates) == 2 and canonical.resolve() != legacy.resolve():
        raise RuntimeError(
            f"Both canonical and legacy bundle directories exist; keep only one: "
            f"{canonical}, {legacy}"
        )
    return candidates[0]


def bundle_paths(
    root: str | Path,
    *,
    dataset: str,
    model: str,
    splits: Iterable[str] = ("train", "valid", "test"),
) -> dict[str, Path]:
    directory = resolve_bundle_directory(root, dataset=dataset, model=model)
    result: dict[str, Path] = {}
    for value in splits:
        split = str(value)
        candidates = [directory / f"{split}.pt"]
        if split == "valid":
            candidates.append(directory / "val.pt")
        existing = [path for path in candidates if path.is_file()]
        if len(existing) > 1:
            raise RuntimeError(
                f"Both valid.pt and val.pt exist in {directory}; keep one"
            )
        result[split] = existing[0] if existing else candidates[0]
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing PT split files: " + ", ".join(missing))
    return result


def evaluation_output_directory(
    root: str | Path,
    *,
    dataset: str,
    model: str,
    task: str,
) -> Path:
    dataset = _safe_component(dataset, "dataset")
    model = _safe_component(model, "model")
    task = _safe_component(task, "task")
    return Path(root).resolve() / dataset / model / task


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def strict_json_dumps(payload: Any, *, indent: int | None = 2) -> str:
    return json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
        allow_nan=False,
    )


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(strict_json_dumps(payload) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
