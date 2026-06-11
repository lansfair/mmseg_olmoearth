from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


PASTIS_S2_10_BANDS = (
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
)


@TRANSFORMS.register_module()
class DINOv3PASTISS2Normalize(BaseTransform):
    """Normalize original PASTIS Sentinel-2 values.

    When ``norm_file`` points to PASTIS-R ``NORM_S2_patch.json``, this transform
    applies per-band mean/std standardization. Otherwise it falls back to the
    course-script style ``x / 10000`` scaling.
    """

    def __init__(
        self,
        scale_factor: float = 10000.0,
        clip: bool = True,
        num_timesteps: int = 12,
        num_bands: int = 10,
        norm_file: str | None = None,
        band_names: tuple[str, ...] = PASTIS_S2_10_BANDS,
        eps: float = 1e-6,
    ) -> None:
        self.scale_factor = scale_factor
        self.clip = clip
        self.num_timesteps = num_timesteps
        self.num_bands = num_bands
        self.norm_file = norm_file
        self.band_names = tuple(band_names)
        self.eps = eps
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None
        if norm_file is not None:
            self.means, self.stds = self._load_band_stats(norm_file)

    @staticmethod
    def _lookup_case_insensitive(payload: dict[str, Any], key: str) -> Any:
        if key in payload:
            return payload[key]
        key_norm = key.upper()
        for candidate, value in payload.items():
            if str(candidate).upper() == key_norm:
                return value
        raise KeyError(key)

    def _extract_stat(self, payload: Any, band_name: str, stat_name: str) -> float:
        if isinstance(payload, dict):
            try:
                band_payload = self._lookup_case_insensitive(payload, band_name)
                if isinstance(band_payload, dict):
                    return float(self._lookup_case_insensitive(band_payload, stat_name))
            except KeyError:
                pass

            for key in (
                stat_name,
                f"{stat_name}s",
                stat_name.upper(),
                f"{stat_name.upper()}S",
            ):
                if key not in payload:
                    continue
                values = payload[key]
                if isinstance(values, dict):
                    return float(self._lookup_case_insensitive(values, band_name))
                if isinstance(values, (list, tuple)):
                    return float(values[self.band_names.index(band_name)])

            for key in (
                f"{band_name}_{stat_name}",
                f"{band_name.lower()}_{stat_name}",
                f"{stat_name}_{band_name}",
                f"{stat_name}_{band_name.lower()}",
            ):
                if key in payload:
                    return float(payload[key])

        raise KeyError(
            f"Could not find {stat_name} for band {band_name} in {self.norm_file}"
        )

    def _load_band_stats(self, norm_file: str) -> tuple[np.ndarray, np.ndarray]:
        path = Path(norm_file)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        means = np.asarray(
            [self._extract_stat(payload, band_name, "mean") for band_name in self.band_names],
            dtype=np.float32,
        )
        stds = np.asarray(
            [self._extract_stat(payload, band_name, "std") for band_name in self.band_names],
            dtype=np.float32,
        )
        if len(means) != self.num_bands or len(stds) != self.num_bands:
            raise ValueError(
                f"Expected {self.num_bands} band stats from {norm_file}, "
                f"got means={len(means)} stds={len(stds)}"
            )
        return means, np.maximum(stds, self.eps)

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        image = results["img"].astype(np.float32, copy=False)
        expected = self.num_timesteps * self.num_bands
        if image.shape[-1] != expected:
            raise ValueError(f"Expected {expected} channels, got {image.shape[-1]}")

        if self.means is not None and self.stds is not None:
            offsets = np.repeat(self.means, self.num_timesteps)
            scales = np.repeat(self.stds, self.num_timesteps)
            image = (image - offsets) / scales
            results["pastis_normalization"] = "norm_s2_patch_mean_std"
        else:
            image = image / self.scale_factor
            if self.clip:
                image = np.clip(image, 0.0, 1.0)
            results["pastis_normalization"] = "scale_10000_clip_0_1"

        results["img"] = image.astype(np.float32, copy=False)
        results["pastis_num_timesteps"] = self.num_timesteps
        results["pastis_num_bands"] = self.num_bands
        return results
