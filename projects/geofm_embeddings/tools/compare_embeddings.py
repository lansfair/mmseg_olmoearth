from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_embedding(path: Path) -> Tensor:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        value = torch.load(path, map_location="cpu")
        if isinstance(value, dict):
            for key in ("embedding", "embeddings", "features", "tensor"):
                if key in value:
                    value = value[key]
                    break
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value)
        return value.float()
    if suffix == ".npy":
        return torch.from_numpy(np.load(path)).float()
    if suffix in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError as exc:
            raise ImportError("GeoTIFF comparison requires rasterio.") from exc
        with rasterio.open(path) as dataset:
            return torch.from_numpy(dataset.read()).float()
    raise ValueError(f"Unsupported embedding file: {path}")


def compare(reference: Tensor, candidate: Tensor) -> dict[str, float | list[int]]:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Shape mismatch: reference={tuple(reference.shape)}, "
            f"candidate={tuple(candidate.shape)}"
        )
    reference = reference.float()
    candidate = candidate.float()
    difference = (reference - candidate).abs()
    if reference.ndim == 1:
        reference_vectors = reference[None]
        candidate_vectors = candidate[None]
    elif reference.ndim == 2:
        reference_vectors = reference
        candidate_vectors = candidate
    elif reference.ndim == 3:
        # Channel-first dense embeddings become one vector per spatial cell.
        reference_vectors = reference.movedim(0, -1).reshape(-1, reference.shape[0])
        candidate_vectors = candidate.movedim(0, -1).reshape(-1, candidate.shape[0])
    elif reference.ndim == 4:
        reference_vectors = reference.movedim(1, -1).reshape(-1, reference.shape[1])
        candidate_vectors = candidate.movedim(1, -1).reshape(-1, candidate.shape[1])
    else:
        raise ValueError(
            f"Unsupported embedding rank for comparison: {reference.ndim}"
        )
    cosine = F.cosine_similarity(reference_vectors, candidate_vectors, dim=-1)
    return {
        "shape": list(reference.shape),
        "mae": difference.mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "max_abs_error": difference.max().item(),
        "mean_cosine_similarity": cosine.mean().item(),
        "min_cosine_similarity": cosine.min().item(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare official and GeoFM adapter embeddings."
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = compare(
        load_embedding(args.reference),
        load_embedding(args.candidate),
    )
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
