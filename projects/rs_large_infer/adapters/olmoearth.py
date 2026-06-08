from __future__ import annotations

from typing import Any

import numpy as np
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.utils import import_modules_from_strings

from ..utils import S2_BANDS, band_indices_for_s2, clean_pipeline, find_transform
from .base import BaseAdapter


class OlmoEarthAdapter(BaseAdapter):
    name = "olmoearth"
    meta_keys = BaseAdapter.meta_keys + (
        "olmoearth_modality",
        "olmoearth_num_timesteps",
        "olmoearth_band_names",
        "present_bands",
        "timestamps",
        "olmoearth_rgb_adapter",
        "olmoearth_raw_band_names",
    )

    @classmethod
    def detect(cls, cfg: Config, raw_pipeline: list[dict[str, Any]]) -> bool:
        return (
            find_transform(raw_pipeline, "RGBToOlmoEarthS2") is not None
            or find_transform(raw_pipeline, "OlmoEarthNormalize") is not None
            or "OlmoEarth" in str(cfg.model.get("type", ""))
            or "OlmoEarth" in str(cfg.model.get("backbone", {}).get("type", ""))
        )

    def __init__(
        self,
        cfg: Config,
        raw_pipeline: list[dict[str, Any]],
        args,
    ) -> None:
        super().__init__(cfg, raw_pipeline, args)
        import_modules_from_strings(
            imports=["projects.olmoearth.olmoearth"],
            allow_failed_imports=False,
        )
        transforms = clean_pipeline(raw_pipeline)
        self.olmo_mode = self._resolve_olmo_mode()
        if self.olmo_mode == "rgb":
            rgb = find_transform(transforms, "RGBToOlmoEarthS2")
            if rgb is None:
                rgb = dict(type="RGBToOlmoEarthS2")
                transforms.append(rgb)
            rgb["rgb_channel_order"] = args.rgb_channel_order or "RGB"
            if args.input_value_range is not None:
                rgb["input_value_range"] = args.input_value_range
            else:
                rgb.setdefault("input_value_range", "auto")
            self.band_indices = [1, 2, 3]
        elif self.olmo_mode == "s2":
            if find_transform(transforms, "OlmoEarthNormalize") is None:
                transforms.append(
                    dict(
                        type="OlmoEarthNormalize",
                        modality="sentinel2_l2a",
                        num_timesteps=cfg.get("num_timesteps", 1),
                    )
                )
            self.to_float32 = args.s2_scale_factor is not None
        self.pipeline = Compose(transforms)

    def _resolve_olmo_mode(self) -> str:
        if self.args.input_mode in {"rgb", "s2"}:
            return self.args.input_mode
        if find_transform(self.raw_pipeline, "RGBToOlmoEarthS2") is not None:
            return "rgb"
        return "s2"

    def prepare(self, src) -> None:
        if self.olmo_mode == "s2":
            self.band_indices = band_indices_for_s2(
                self.args.source_band_names,
                src.count,
            )
        elif src.count < 3:
            raise ValueError(f"RGB mode requires at least 3 bands, got {src.count}")

    def read(
        self,
        src,
        grid: tuple[int, int, int, int, int, int, int, int],
    ) -> np.ndarray:
        image = super().read(src, grid)
        if self.olmo_mode == "s2" and self.args.s2_scale_factor is not None:
            image = image.astype(np.float32, copy=False) * self.args.s2_scale_factor
        return image

    def make_results(
        self,
        image: np.ndarray,
        src,
        grid: tuple[int, int, int, int, int, int, int, int],
    ) -> dict[str, Any]:
        results = super().make_results(image, src, grid)
        results["timestamps"] = np.asarray([tuple(self.args.timestamp)], dtype=np.int64)
        if self.olmo_mode == "s2":
            results["olmoearth_modality"] = "sentinel2_l2a"
            results["olmoearth_num_timesteps"] = 1
            results["olmoearth_band_names"] = S2_BANDS
            results["present_bands"] = S2_BANDS
        return results
