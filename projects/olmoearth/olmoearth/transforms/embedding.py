from __future__ import annotations

from typing import Any

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


def _read_geotiff(path: str) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        return src.read()


@TRANSFORMS.register_module()
class LoadOlmoEarthEmbedding(BaseTransform):
    """Load precomputed dense OLMoEarth embeddings for linear probing.

    The embedding GeoTIFF is stored as CHW and converted to HWC so it can be
    packed by ``PackOlmoEarthSegInputs`` like a normal MMSeg image tensor.
    Labels stay at original pixel resolution; ``OlmoEarthPatchLinearHead``
    expands one embedding token to ``patch_size x patch_size`` logits.
    """

    def __init__(
        self,
        ignore_index: int = 255,
        dtype: str = "float32",
    ) -> None:
        self.ignore_index = ignore_index
        self.dtype = dtype

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        embedding_path = results.get("embedding_path")
        if embedding_path is None:
            raise KeyError("LoadOlmoEarthEmbedding requires 'embedding_path'.")
        if not str(embedding_path).lower().endswith((".tif", ".tiff")):
            raise ValueError(
                "LoadOlmoEarthEmbedding only supports GeoTIFF embeddings, "
                f"got {embedding_path}"
            )

        embedding = _read_geotiff(embedding_path).astype(self.dtype, copy=False)
        label = _read_geotiff(results["seg_map_path"])[0].astype(
            np.int64,
            copy=False,
        )
        results["img"] = np.moveaxis(embedding, 0, -1)
        results["gt_seg_map"] = label
        results["ori_shape"] = label.shape
        results["img_shape"] = results["img"].shape[:2]

        valid_mask_path = results.get("valid_mask_path")
        if valid_mask_path:
            results["gt_valid_mask"] = _read_geotiff(valid_mask_path)[0].astype(
                np.float32,
                copy=False,
            )
        return results
