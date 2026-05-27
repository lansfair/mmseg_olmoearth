from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


def _load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() in {".pt", ".pth"}:
        value = torch.load(path, map_location="cpu")
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)
    raise ValueError(f"Unsupported array file suffix: {path}")


def _to_hw_flat(image: np.ndarray, layout: str) -> np.ndarray:
    layout = layout.upper()
    if layout == "HWC":
        return image
    if layout == "HWTC":
        h, w, t, c = image.shape
        return image.transpose(0, 1, 3, 2).reshape(h, w, c * t)
    if layout == "HWCT":
        h, w, c, t = image.shape
        return image.reshape(h, w, c * t)
    if layout == "TCHW":
        t, c, h, w = image.shape
        return image.transpose(2, 3, 1, 0).reshape(h, w, c * t)
    if layout == "CTHW":
        c, t, h, w = image.shape
        return image.transpose(2, 3, 0, 1).reshape(h, w, c * t)
    raise ValueError(f"Unsupported OLMoEarth image layout: {layout}")


@TRANSFORMS.register_module()
class LoadOlmoEarthArrays(BaseTransform):
    """Load OLMoEarth image, label, optional mask and timestamps."""

    def __init__(
        self,
        image_layout: str = "TCHW",
        ignore_index: int = 255,
        source_ignore_values: tuple[int, ...] = (-1,),
        reduce_zero_label: bool = False,
    ) -> None:
        self.image_layout = image_layout
        self.ignore_index = ignore_index
        self.source_ignore_values = source_ignore_values
        self.reduce_zero_label = reduce_zero_label

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        results["seg_fields"] = ["gt_seg_map"]
        image = _load_array(results["img_path"]).astype(np.float32, copy=False)
        results["img"] = _to_hw_flat(image, self.image_layout)
        results["img_shape"] = results["img"].shape[:2]
        results["ori_shape"] = results["img"].shape[:2]

        label = _load_array(results["seg_map_path"]).squeeze().astype(np.int64)
        if self.reduce_zero_label:
            label = label.copy()
            label[label == 0] = self.ignore_index
            label = label - 1
            label[label == self.ignore_index - 1] = self.ignore_index
        if self.source_ignore_values:
            label = label.copy()
            for value in self.source_ignore_values:
                label[label == value] = self.ignore_index
        results["gt_seg_map"] = label

        valid_mask_path = results.get("valid_mask_path")
        if valid_mask_path is not None:
            valid = _load_array(valid_mask_path).squeeze().astype(np.float32)
            results["gt_valid_mask"] = valid
            results["seg_fields"].append("gt_valid_mask")

        timestamps_path = results.get("timestamps_path")
        if timestamps_path is not None:
            results["timestamps"] = _load_array(timestamps_path).astype(
                np.int64
            )
        elif "timestamps" in results:
            results["timestamps"] = np.asarray(
                results["timestamps"],
                dtype=np.int64,
            )

        return results
