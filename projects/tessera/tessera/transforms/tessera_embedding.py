from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


def _read_array(path: str) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix in {".tif", ".tiff"}:
        import rasterio

        with rasterio.open(path) as src:
            return src.read()
    raise ValueError(f"Unsupported array format for {path}")


def _to_hwc(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D embedding array, got {array.shape}")
    # GeoTIFF is normally CHW; TESSERA npy output is normally HWC.
    if array.shape[0] in {128, 192} and array.shape[-1] not in {128, 192}:
        return np.moveaxis(array, 0, -1)
    return array


def _read_label(path: str) -> np.ndarray:
    label = _read_array(path)
    if label.ndim == 3:
        if label.shape[0] == 1:
            label = label[0]
        elif label.shape[-1] == 1:
            label = label[..., 0]
        else:
            raise ValueError(
                f"Expected a single-channel label map, got {label.shape}"
            )
    return label.astype(np.int64, copy=False)


@TRANSFORMS.register_module()
class LoadTesseraEmbedding(BaseTransform):
    """Load precomputed TESSERA embeddings and segmentation labels.

    Supports fp32 ``.npy``/GeoTIFF embeddings and v1.1 QAT outputs saved as
    ``*_emb128_int8.npy`` plus a matching per-pixel scale array.
    """

    def __init__(
        self,
        ignore_index: int = 255,
        dtype: str = "float32",
        embedding_dim: int = 128,
        dequantize: bool = True,
    ) -> None:
        self.ignore_index = ignore_index
        self.dtype = dtype
        self.embedding_dim = embedding_dim
        self.dequantize = dequantize

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        embedding_path = results.get("embedding_path")
        if embedding_path is None:
            raise KeyError("LoadTesseraEmbedding requires 'embedding_path'.")

        embedding = _to_hwc(_read_array(embedding_path))
        scales_path = results.get("scales_path") or results.get("scale_path")
        if scales_path is not None and self.dequantize:
            scales = _read_array(scales_path).astype(np.float32, copy=False)
            if scales.ndim == 3:
                scales = np.squeeze(scales)
            embedding = embedding.astype(np.float32, copy=False)
            embedding = embedding * scales[..., None]

        embedding = embedding.astype(self.dtype, copy=False)
        if embedding.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected TESSERA embedding dim {self.embedding_dim}, "
                f"got {embedding.shape[-1]} from {embedding_path}"
            )

        label = _read_label(results["seg_map_path"])
        results["img"] = embedding
        results["gt_seg_map"] = label
        results["ori_shape"] = label.shape
        results["img_shape"] = embedding.shape[:2]

        valid_mask_path = results.get("valid_mask_path")
        if valid_mask_path:
            valid_mask = _read_array(valid_mask_path)
            if valid_mask.ndim == 3:
                valid_mask = np.squeeze(valid_mask)
            results["gt_valid_mask"] = valid_mask.astype(
                np.float32,
                copy=False,
            )
        return results
