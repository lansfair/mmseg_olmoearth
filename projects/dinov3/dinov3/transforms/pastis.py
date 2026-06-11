from __future__ import annotations

from typing import Any

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class DINOv3PASTISS2Normalize(BaseTransform):
    """Normalize original PASTIS Sentinel-2 values like the course script."""

    def __init__(
        self,
        scale_factor: float = 10000.0,
        clip: bool = True,
        num_timesteps: int = 12,
        num_bands: int = 10,
    ) -> None:
        self.scale_factor = scale_factor
        self.clip = clip
        self.num_timesteps = num_timesteps
        self.num_bands = num_bands

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        image = results["img"].astype(np.float32, copy=False)
        expected = self.num_timesteps * self.num_bands
        if image.shape[-1] != expected:
            raise ValueError(f"Expected {expected} channels, got {image.shape[-1]}")
        image = image / self.scale_factor
        if self.clip:
            image = np.clip(image, 0.0, 1.0)
        results["img"] = image.astype(np.float32, copy=False)
        results["pastis_num_timesteps"] = self.num_timesteps
        results["pastis_num_bands"] = self.num_bands
        return results
