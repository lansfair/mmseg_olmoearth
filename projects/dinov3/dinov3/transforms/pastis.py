from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class DINOv3PASTISS2Normalize(BaseTransform):
    """Normalize original PASTIS Sentinel-2 values.

    ``NORM_S2_patch.json`` stores per-fold 10-band statistics as
    ``Fold_1`` ... ``Fold_5``. When ``norm_file`` is set, this transform uses
    the average statistics from ``folds``. Use training folds for validation
    and testing too, to avoid leaking evaluation split statistics.
    """

    def __init__(
        self,
        scale_factor: float = 10000.0,
        clip: bool = True,
        num_timesteps: int = 12,
        num_bands: int = 10,
        norm_file: str | None = None,
        folds: tuple[int, ...] = (1, 2, 3),
        eps: float = 1e-6,
    ) -> None:
        self.scale_factor = scale_factor
        self.clip = clip
        self.num_timesteps = num_timesteps
        self.num_bands = num_bands
        self.norm_file = norm_file
        self.folds = tuple(int(fold) for fold in folds)
        self.eps = eps
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None
        if norm_file is not None:
            self.means, self.stds = self._load_band_stats(norm_file)

    def _fold_key(self, fold: int) -> str:
        key = f"Fold_{fold}"
        if key:
            return key
        raise ValueError(f"Invalid fold: {fold}")

    def _load_fold_stat(
        self,
        payload: dict[str, Any],
        fold: int,
        stat_name: str,
    ) -> np.ndarray:
        key = self._fold_key(fold)
        if key not in payload:
            raise KeyError(f"{key} not found in {self.norm_file}")
        if stat_name not in payload[key]:
            raise KeyError(f"{key}.{stat_name} not found in {self.norm_file}")
        values = np.asarray(payload[key][stat_name], dtype=np.float32)
        if values.shape != (self.num_bands,):
            raise ValueError(
                f"Expected {self.num_bands} values for {key}.{stat_name}, "
                f"got shape {values.shape}"
            )
        return values

    def _load_band_stats(self, norm_file: str) -> tuple[np.ndarray, np.ndarray]:
        path = Path(norm_file)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        means = np.stack(
            [self._load_fold_stat(payload, fold, "mean") for fold in self.folds],
            axis=0,
        )
        stds = np.stack(
            [self._load_fold_stat(payload, fold, "std") for fold in self.folds],
            axis=0,
        )
        return means.mean(axis=0), np.maximum(stds.mean(axis=0), self.eps)

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        image = results["img"].astype(np.float32, copy=False)
        expected = self.num_timesteps * self.num_bands
        if image.shape[-1] != expected:
            raise ValueError(f"Expected {expected} channels, got {image.shape[-1]}")

        if self.means is not None and self.stds is not None:
            offsets = np.repeat(self.means, self.num_timesteps)
            scales = np.repeat(self.stds, self.num_timesteps)
            image = (image - offsets) / scales
            results["pastis_normalization"] = "norm_s2_patch_train_fold_mean_std"
        else:
            image = image / self.scale_factor
            if self.clip:
                image = np.clip(image, 0.0, 1.0)
            results["pastis_normalization"] = "scale_10000_clip_0_1"

        results["img"] = image.astype(np.float32, copy=False)
        results["pastis_num_timesteps"] = self.num_timesteps
        results["pastis_num_bands"] = self.num_bands
        return results
