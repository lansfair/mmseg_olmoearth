from __future__ import annotations

from typing import Any

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS

from ..utils import RGB_TO_SENTINEL2_L2A, get_modality_bands


def _load_computed_norm(modality: str) -> dict[str, dict[str, float]]:
    from olmoearth_pretrain.data.normalize import load_computed_config

    return load_computed_config()[modality]


@TRANSFORMS.register_module()
class OlmoEarthNormalize(BaseTransform):
    """Apply OLMoEarth computed 2-std normalization to flattened imagery."""

    def __init__(
        self,
        modality: str,
        num_timesteps: int = 12,
        band_names: list[str] | None = None,
        std_multiplier: float = 2.0,
    ) -> None:
        self.modality = modality
        self.num_timesteps = num_timesteps
        self.band_names = band_names or list(get_modality_bands(modality))
        self.std_multiplier = std_multiplier
        self.norm_config = _load_computed_norm(modality)

    def _normalize_band(
        self,
        values: np.ndarray,
        band_name: str,
    ) -> np.ndarray:
        stats = self.norm_config[band_name]
        min_val = stats["mean"] - self.std_multiplier * stats["std"]
        max_val = stats["mean"] + self.std_multiplier * stats["std"]
        return (values - min_val) / (max_val - min_val)

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        image = results["img"].astype(np.float32, copy=True)
        expected = len(self.band_names) * self.num_timesteps
        if image.shape[-1] != expected:
            raise ValueError(
                f"Expected {expected} channels, got {image.shape[-1]}"
            )
        for band_idx, band_name in enumerate(self.band_names):
            for t in range(self.num_timesteps):
                channel_idx = band_idx * self.num_timesteps + t
                image[..., channel_idx] = self._normalize_band(
                    image[..., channel_idx], band_name
                )
        results["img"] = image
        results["olmoearth_modality"] = self.modality
        results["olmoearth_num_timesteps"] = self.num_timesteps
        results["olmoearth_band_names"] = self.band_names
        results.setdefault("present_bands", self.band_names)
        return results


def _normalization_bounds(
    method: str,
    means: np.ndarray,
    stds: np.ndarray,
    mins: np.ndarray | None = None,
    maxs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "norm_no_clip_2_std":
        return means - 2.0 * stds, means + 2.0 * stds
    if method == "norm_no_clip":
        return means - stds, means + stds
    if method == "standardize":
        return means, stds
    if method == "norm_yes_clip_min_max_int":
        if mins is None or maxs is None:
            raise ValueError(
                "mins/maxs are required for norm_yes_clip_min_max_int"
            )
        return mins, maxs
    raise ValueError(f"Unsupported dataset normalization method: {method}")


def _load_mados_stats() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
]:
    try:
        from olmoearth_pretrain.evals.datasets.constants import (
            EVAL_S2_BAND_NAMES,
            EVAL_TO_OLMOEARTH_S2_BANDS,
        )
        from olmoearth_pretrain.evals.datasets.mados_dataset import BAND_STATS
        from olmoearth_pretrain.evals.datasets.utils import load_min_max_stats

        minmax = load_min_max_stats()["mados"]["sentinel2_l2a"]
        means = np.asarray(
            [
                BAND_STATS[band_name]["mean"]
                for band_name in EVAL_S2_BAND_NAMES
            ],
            dtype=np.float32,
        )
        stds = np.asarray(
            [BAND_STATS[band_name]["std"] for band_name in EVAL_S2_BAND_NAMES],
            dtype=np.float32,
        )
        mins = np.asarray(
            [minmax[band_name]["min"] for band_name in EVAL_S2_BAND_NAMES],
            dtype=np.float32,
        )
        maxs = np.asarray(
            [minmax[band_name]["max"] for band_name in EVAL_S2_BAND_NAMES],
            dtype=np.float32,
        )
        indices = np.asarray(EVAL_TO_OLMOEARTH_S2_BANDS, dtype=np.int64)
        return means[indices], stds[indices], mins[indices], maxs[indices]
    except Exception as exc:
        raise RuntimeError(
            "MADOS dataset-specific normalization requires olmoearth_pretrain "
            "on PYTHONPATH so the exact reference stats can be reused."
        ) from exc


@TRANSFORMS.register_module()
class OlmoEarthDatasetNormalize(BaseTransform):
    """Apply dataset-specific normalization used by OLMoEarth eval datasets."""

    def __init__(
        self,
        dataset_name: str,
        modality: str,
        num_timesteps: int = 1,
        method: str = "norm_no_clip_2_std",
    ) -> None:
        self.dataset_name = dataset_name
        self.modality = modality
        self.num_timesteps = num_timesteps
        self.method = method
        if dataset_name == "mados" and modality == "sentinel2_l2a":
            self.means, self.stds, self.mins, self.maxs = _load_mados_stats()
        else:
            raise ValueError(
                "Unsupported dataset-specific normalization: "
                f"{dataset_name}/{modality}"
            )

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        image = results["img"].astype(np.float32, copy=True)
        num_bands = len(self.means)
        expected = num_bands * self.num_timesteps
        if image.shape[-1] != expected:
            raise ValueError(
                f"Expected {expected} channels, got {image.shape[-1]}"
            )

        mins_or_means, maxs_or_stds = _normalization_bounds(
            self.method,
            self.means,
            self.stds,
            self.mins,
            self.maxs,
        )
        for band_idx in range(num_bands):
            for t in range(self.num_timesteps):
                channel_idx = band_idx * self.num_timesteps + t
                if self.method == "standardize":
                    image[..., channel_idx] = (
                        image[..., channel_idx] - mins_or_means[band_idx]
                    ) / maxs_or_stds[band_idx]
                else:
                    image[..., channel_idx] = (
                        image[..., channel_idx] - mins_or_means[band_idx]
                    ) / (maxs_or_stds[band_idx] - mins_or_means[band_idx])
        results["img"] = image
        results["olmoearth_modality"] = self.modality
        results["olmoearth_num_timesteps"] = self.num_timesteps
        results["olmoearth_band_names"] = list(
            get_modality_bands(self.modality)
        )
        results.setdefault("present_bands", results["olmoearth_band_names"])
        return results


@TRANSFORMS.register_module()
class RGBToOlmoEarthS2(BaseTransform):
    """Explicit RGB compatibility adapter.

    RGB is mapped to Sentinel-2 B04/B03/B02 and missing Sentinel-2 bands are
    filled with normalized zero. This is a compatibility path, not a paper
    reproduction path.
    """

    def __init__(
        self,
        num_timesteps: int = 1,
        rgb_channel_order: str = "RGB",
        input_value_range: str = "auto",
        std_multiplier: float = 2.0,
    ) -> None:
        rgb_channel_order = rgb_channel_order.upper()
        if sorted(rgb_channel_order) != ["B", "G", "R"]:
            raise ValueError("rgb_channel_order must be a permutation of RGB")
        if input_value_range not in {"auto", "0_255", "0_1", "s2"}:
            raise ValueError(
                "input_value_range must be auto, 0_255, 0_1, or s2"
            )
        self.num_timesteps = num_timesteps
        self.rgb_channel_order = rgb_channel_order
        self.input_value_range = input_value_range
        self.band_names = list(get_modality_bands("sentinel2_l2a"))
        self.norm_config = _load_computed_norm("sentinel2_l2a")
        self.std_multiplier = std_multiplier

    def _to_s2_scale(self, image: np.ndarray) -> np.ndarray:
        if self.input_value_range == "s2":
            return image
        mode = self.input_value_range
        if mode == "auto":
            max_value = float(np.nanmax(image)) if image.size else 0.0
            mode = "0_1" if max_value <= 1.5 else "0_255"
        if mode == "0_1":
            return image * 10000.0
        if mode == "0_255":
            return image * (10000.0 / 255.0)
        return image

    def _normalize_band(
        self,
        values: np.ndarray,
        band_name: str,
    ) -> np.ndarray:
        stats = self.norm_config[band_name]
        min_val = stats["mean"] - self.std_multiplier * stats["std"]
        max_val = stats["mean"] + self.std_multiplier * stats["std"]
        return (values - min_val) / (max_val - min_val)

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        image = results["img"].astype(np.float32, copy=False)
        expected = 3 * self.num_timesteps
        if image.ndim != 3 or image.shape[-1] != expected:
            raise ValueError(
                f"Expected RGB image with {expected} channels, "
                f"got {image.shape}"
            )
        image = self._to_s2_scale(image)
        h, w = image.shape[:2]
        out = np.zeros(
            (h, w, len(self.band_names) * self.num_timesteps),
            dtype=np.float32,
        )
        channel_to_index = {
            name: idx for idx, name in enumerate(self.rgb_channel_order)
        }
        for t in range(self.num_timesteps):
            rgb_base = t * 3
            for rgb_name, s2_band in RGB_TO_SENTINEL2_L2A.items():
                rgb_idx = rgb_base + channel_to_index[rgb_name]
                band_idx = self.band_names.index(s2_band)
                out_idx = band_idx * self.num_timesteps + t
                out[..., out_idx] = self._normalize_band(
                    image[..., rgb_idx],
                    s2_band,
                )
        results["img"] = out
        results["olmoearth_modality"] = "sentinel2_l2a"
        results["olmoearth_num_timesteps"] = self.num_timesteps
        results["olmoearth_band_names"] = self.band_names
        results["present_bands"] = list(RGB_TO_SENTINEL2_L2A.values())
        results["olmoearth_rgb_adapter"] = {
            "rgb_channel_order": self.rgb_channel_order,
            "input_value_range": self.input_value_range,
            "mapped_bands": RGB_TO_SENTINEL2_L2A,
        }
        return results
