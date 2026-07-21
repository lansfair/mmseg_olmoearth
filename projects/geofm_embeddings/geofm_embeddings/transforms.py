from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS

from projects.olmoearth.olmoearth.transforms.normalize import (
    _load_computed_norm,
)
from projects.olmoearth.olmoearth.utils import (
    RGB_TO_SENTINEL2_L2A,
    get_modality_bands,
)


@TRANSFORMS.register_module()
class ResizeImageOnly(BaseTransform):
    """Resize imagery while retaining the original dense annotation.

    Embeddings are extracted from a common, compact model input grid.  The
    512 x 512 Potsdam label is deliberately left untouched so downstream
    dense probes can upsample features to the original target resolution.
    """

    def __init__(self, size: int = 64) -> None:
        if size < 1:
            raise ValueError("size must be positive")
        self.size = int(size)

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        image = results["img"]
        results["img"] = cv2.resize(
            image,
            (self.size, self.size),
            interpolation=cv2.INTER_LINEAR,
        )
        if results["img"].ndim == 2:
            results["img"] = results["img"][..., None]
        results["img_shape"] = results["img"].shape[:2]
        results["geofm_model_input_shape"] = (self.size, self.size)
        return results


@TRANSFORMS.register_module()
class RGBToGeoFMS2(BaseTransform):
    """Map Potsdam BGR/RGB pixels to a 12-band Sentinel-2 proxy.

    ``representation='normalized'`` reproduces the OLMoEarth 2-standard-
    deviation scaling. ``representation='reflectance'`` emits 0..10000
    reflectance-like values for wrappers that apply their own normalization.
    Missing non-RGB bands are filled with their training-set mean.
    """

    def __init__(
        self,
        rgb_channel_order: str = "BGR",
        input_value_range: str = "0_255",
        representation: str = "normalized",
        std_multiplier: float = 2.0,
    ) -> None:
        rgb_channel_order = rgb_channel_order.upper()
        if sorted(rgb_channel_order) != ["B", "G", "R"]:
            raise ValueError("rgb_channel_order must be a permutation of RGB")
        if input_value_range not in {"0_255", "0_1", "s2"}:
            raise ValueError("input_value_range must be 0_255, 0_1, or s2")
        if representation not in {"normalized", "reflectance"}:
            raise ValueError("representation must be normalized or reflectance")
        self.rgb_channel_order = rgb_channel_order
        self.input_value_range = input_value_range
        self.representation = representation
        self.std_multiplier = float(std_multiplier)
        self.band_names = list(get_modality_bands("sentinel2_l2a"))
        self.norm_config = _load_computed_norm("sentinel2_l2a")

    def _reflectance(self, image: np.ndarray) -> np.ndarray:
        if self.input_value_range == "0_255":
            return image * (10000.0 / 255.0)
        if self.input_value_range == "0_1":
            return image * 10000.0
        return image

    def _normalize(self, values: np.ndarray, band: str) -> np.ndarray:
        stats = self.norm_config[band]
        minimum = stats["mean"] - self.std_multiplier * stats["std"]
        maximum = stats["mean"] + self.std_multiplier * stats["std"]
        return (values - minimum) / (maximum - minimum)

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        image = results["img"].astype(np.float32, copy=False)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected an HWC RGB image, got {image.shape}")
        image = self._reflectance(image)
        height, width = image.shape[:2]
        means = np.asarray(
            [self.norm_config[name]["mean"] for name in self.band_names],
            dtype=np.float32,
        )
        output = np.broadcast_to(
            means[None, None, :], (height, width, len(means))
        ).copy()
        channel_indices = {
            name: index for index, name in enumerate(self.rgb_channel_order)
        }
        for rgb_name, s2_band in RGB_TO_SENTINEL2_L2A.items():
            output[..., self.band_names.index(s2_band)] = image[
                ..., channel_indices[rgb_name]
            ]
        if self.representation == "normalized":
            for index, band in enumerate(self.band_names):
                output[..., index] = self._normalize(output[..., index], band)
        results["img"] = output
        results["olmoearth_modality"] = "sentinel2_l2a"
        results["olmoearth_num_timesteps"] = 1
        results["olmoearth_band_names"] = self.band_names
        results["present_bands"] = list(RGB_TO_SENTINEL2_L2A.values())
        results["geofm_input_representation"] = self.representation
        return results
