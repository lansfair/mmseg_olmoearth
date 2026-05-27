from __future__ import annotations

from typing import Any

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


def _seg_fields(results: dict[str, Any]) -> list[str]:
    return list(results.get("seg_fields", []))


def _apply_seg_fields(results: dict[str, Any], fn) -> None:
    for field in _seg_fields(results):
        if field in results:
            results[field] = fn(results[field])


def _pad_hw(
    array: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
    value: float | int,
) -> np.ndarray:
    if array.ndim == 2:
        pad_width = ((top, bottom), (left, right))
    else:
        pad_width = ((top, bottom), (left, right)) + tuple(
            (0, 0) for _ in range(array.ndim - 2)
        )
    return np.pad(
        array,
        pad_width=pad_width,
        mode="constant",
        constant_values=value,
    )


def _crop_hw(array: np.ndarray, top: int, left: int, size: int) -> np.ndarray:
    return array[top : top + size, left : left + size, ...]


@TRANSFORMS.register_module()
class OlmoEarthPad(BaseTransform):
    """Pad or center-crop OLMoEarth arrays to match rslearn Pad behavior."""

    def __init__(
        self,
        size: int,
        mode: str = "center",
        img_pad_value: float = 0.0,
        seg_pad_value: int = 0,
        valid_pad_value: float = 0.0,
    ) -> None:
        if mode not in {"center", "topleft"}:
            raise ValueError("mode must be 'center' or 'topleft'")
        self.size = size
        self.mode = mode
        self.img_pad_value = img_pad_value
        self.seg_pad_value = seg_pad_value
        self.valid_pad_value = valid_pad_value

    def _offsets(self, length: int) -> tuple[int, int, int, int]:
        extra = self.size - length
        if self.mode == "center":
            before = extra // 2
            after = extra - before
        else:
            before = 0
            after = extra
        crop_before = max(-before, 0)
        crop_after = length - max(-after, 0)
        pad_before = max(before, 0)
        pad_after = max(after, 0)
        return crop_before, crop_after, pad_before, pad_after

    def _apply(self, array: np.ndarray, value: float | int) -> np.ndarray:
        h0, h1, top, bottom = self._offsets(array.shape[0])
        w0, w1, left, right = self._offsets(array.shape[1])
        cropped = array[h0:h1, w0:w1, ...]
        return _pad_hw(cropped, top, bottom, left, right, value)

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        results["img"] = self._apply(results["img"], self.img_pad_value)
        if "gt_seg_map" in results:
            results["gt_seg_map"] = self._apply(
                results["gt_seg_map"],
                self.seg_pad_value,
            )
        if "gt_valid_mask" in results:
            results["gt_valid_mask"] = self._apply(
                results["gt_valid_mask"],
                self.valid_pad_value,
            )
        results["img_shape"] = results["img"].shape[:2]
        return results


@TRANSFORMS.register_module()
class OlmoEarthCrop(BaseTransform):
    """Crop OLMoEarth arrays with rslearn-style random or center offsets."""

    def __init__(self, crop_size: int, mode: str = "random") -> None:
        if mode not in {"center", "random"}:
            raise ValueError("mode must be 'center' or 'random'")
        self.crop_size = crop_size
        self.mode = mode

    def _offsets(self, height: int, width: int) -> tuple[int, int]:
        max_top = height - self.crop_size
        max_left = width - self.crop_size
        if max_top < 0 or max_left < 0:
            raise ValueError(
                f"crop_size={self.crop_size} exceeds image size "
                f"{(height, width)}"
            )
        if self.mode == "center":
            return max_top // 2, max_left // 2
        return (
            np.random.randint(0, max_top + 1),
            np.random.randint(0, max_left + 1),
        )

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        top, left = self._offsets(*results["img"].shape[:2])
        results["img"] = _crop_hw(
            results["img"],
            top,
            left,
            self.crop_size,
        ).copy()
        _apply_seg_fields(
            results,
            lambda x: _crop_hw(x, top, left, self.crop_size).copy(),
        )
        results["img_shape"] = results["img"].shape[:2]
        return results


@TRANSFORMS.register_module()
class OlmoEarthRandomFlip(BaseTransform):
    """Independent horizontal and vertical flips for array-based samples."""

    def __init__(self, horizontal: bool = True, vertical: bool = True) -> None:
        self.horizontal = horizontal
        self.vertical = vertical

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        if self.horizontal and np.random.randint(0, 2) == 0:
            results["img"] = np.flip(results["img"], axis=1).copy()
            _apply_seg_fields(results, lambda x: np.flip(x, axis=1).copy())
        if self.vertical and np.random.randint(0, 2) == 0:
            results["img"] = np.flip(results["img"], axis=0).copy()
            _apply_seg_fields(results, lambda x: np.flip(x, axis=0).copy())
        return results
