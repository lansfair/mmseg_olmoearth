from __future__ import annotations

from typing import Any

import numpy as np
from mmengine.config import Config
from mmengine.dataset import Compose

from ..utils import (
    as_float_list,
    as_int_list,
    clean_pipeline,
    find_transform,
    parse_days_from_date,
    parse_days_from_filename,
    window_lon_lat,
)
from .base import BaseAdapter


class CopernicusAdapter(BaseAdapter):
    """CopernicusFM 适配器：为每个滑窗生成 lon/lat/time/area 元信息。"""

    name = "copernicus"
    meta_keys = BaseAdapter.meta_keys + ("copernicus_meta",)

    @classmethod
    def detect(cls, cfg: Config, raw_pipeline: list[dict[str, Any]]) -> bool:
        """根据模型、backbone 或 transform 名称判断是否为 CopernicusFM 配置。"""

        model_type = str(cfg.model.get("type", ""))
        backbone_type = str(cfg.model.get("backbone", {}).get("type", ""))
        transform_types = {str(transform.get("type")) for transform in raw_pipeline}
        return (
            "Copernicus" in model_type
            or "Copernicus" in backbone_type
            or "LoadCopernicusGeoTiffImageFromFile" in transform_types
            or "AddCopernicusMeta" in transform_types
        )

    def __init__(
        self,
        cfg: Config,
        raw_pipeline: list[dict[str, Any]],
        args,
    ) -> None:
        """初始化 Copernicus 读图参数、时间信息和空间元信息来源。"""

        super().__init__(cfg, raw_pipeline, args)
        self.pipeline = Compose(clean_pipeline(raw_pipeline, {"AddCopernicusMeta"}))
        self.loader = find_transform(raw_pipeline, "LoadCopernicusGeoTiffImageFromFile")
        self.to_float32 = True
        self.nan_to_num = True
        if self.loader is not None:
            self.band_indices = self.band_indices or as_int_list(
                self.loader.get("band_indices")
            )
            self.band_scales = self.band_scales or as_float_list(
                self.loader.get("band_scales")
            )
            self.nan_to_num = bool(self.loader.get("nan_to_num", True))
            self.to_float32 = bool(self.loader.get("to_float32", True))
        self.patch_area = self._get_patch_area()
        self.sensing_time = self._get_sensing_time()
        self.lon_lat_override = (
            tuple(args.copernicus_lon_lat)
            if args.copernicus_lon_lat is not None
            else None
        )

    def _get_patch_area(self) -> float:
        """按命令行、pipeline、backbone 的优先级解析 patch_area。"""

        if self.args.copernicus_patch_area is not None:
            return float(self.args.copernicus_patch_area)
        if self.loader is not None and self.loader.get("patch_area") is not None:
            return float(self.loader["patch_area"])
        add_meta = find_transform(self.raw_pipeline, "AddCopernicusMeta")
        if add_meta is not None and add_meta.get("patch_area") is not None:
            return float(add_meta["patch_area"])
        patch_area = self.cfg.model.get("backbone", {}).get("patch_area")
        if patch_area is not None:
            return float(patch_area)
        return float("nan")

    def _get_sensing_time(self) -> float:
        """解析 sensing time，返回 days since 1970-01-01。"""

        if self.args.copernicus_date_days is not None:
            return float(self.args.copernicus_date_days)
        parsed = parse_days_from_date(self.args.copernicus_date)
        if not np.isnan(parsed):
            return parsed
        if self.loader is None:
            return float("nan")
        return parse_days_from_filename(
            self.args.image,
            self.loader.get("date_separator"),
            int(self.loader.get("date_token_index", 1)),
        )

    def make_results(
        self,
        image: np.ndarray,
        src,
        grid: tuple[int, int, int, int, int, int, int, int],
    ) -> dict[str, Any]:
        """向 results 中追加 CopernicusFM 需要的 copernicus_meta。"""

        results = super().make_results(image, src, grid)
        lon, lat = window_lon_lat(src, grid, self.lon_lat_override)
        results["copernicus_meta"] = np.array(
            [lon, lat, self.sensing_time, self.patch_area],
            dtype=np.float32,
        )
        return results
