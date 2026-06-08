from __future__ import annotations

from mmengine.config import Config

from ..utils import uses_olmoearth_feature_backbone
from .base import BaseAdapter
from .copernicus import CopernicusAdapter
from .olmoearth import OlmoEarthAdapter


def select_adapter(
    cfg: Config,
    raw_pipeline: list[dict[str, Any]],
    args,
) -> BaseAdapter:
    if uses_olmoearth_feature_backbone(cfg):
        if args.input_mode in {"rgb", "s2", "copernicus"}:
            raise ValueError(
                "OlmoEarthFeatureBackbone configs are offline embedding "
                "configs. They cannot run raw image adapters "
                f"(--input-mode {args.input_mode})."
            )
        return BaseAdapter(cfg, raw_pipeline, args)
    if args.input_mode == "copernicus":
        return CopernicusAdapter(cfg, raw_pipeline, args)
    if args.input_mode in {"rgb", "s2"} and OlmoEarthAdapter.detect(cfg, raw_pipeline):
        return OlmoEarthAdapter(cfg, raw_pipeline, args)
    if args.input_mode == "auto":
        if CopernicusAdapter.detect(cfg, raw_pipeline):
            return CopernicusAdapter(cfg, raw_pipeline, args)
        if OlmoEarthAdapter.detect(cfg, raw_pipeline):
            return OlmoEarthAdapter(cfg, raw_pipeline, args)
    return BaseAdapter(cfg, raw_pipeline, args)


__all__ = [
    "BaseAdapter",
    "CopernicusAdapter",
    "OlmoEarthAdapter",
    "select_adapter",
]
