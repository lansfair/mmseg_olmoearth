from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmengine.dataset import BaseDataset
from mmseg.registry import DATASETS

from ..utils import (
    AWF_CLASSES,
    AWF_PALETTE,
    CROP_TYPE_CLASSES,
    CROP_TYPE_PALETTE,
    MADOS_CLASSES,
    MADOS_PALETTE,
    NANDI_CLASSES,
    NANDI_PALETTE,
    PASTIS_CLASSES,
    PASTIS_PALETTE,
    POTSDAM_CLASSES,
    POTSDAM_PALETTE,
    SEN1FLOODS11_CLASSES,
    SEN1FLOODS11_PALETTE,
)


DATASET_METAINFO = {
    "pastis": {"classes": PASTIS_CLASSES, "palette": PASTIS_PALETTE},
    "sen1floods11": {
        "classes": SEN1FLOODS11_CLASSES,
        "palette": SEN1FLOODS11_PALETTE,
    },
    "mados": {"classes": MADOS_CLASSES, "palette": MADOS_PALETTE},
    "awf": {"classes": AWF_CLASSES, "palette": AWF_PALETTE},
    "nandi": {"classes": NANDI_CLASSES, "palette": NANDI_PALETTE},
    "potsdam": {"classes": POTSDAM_CLASSES, "palette": POTSDAM_PALETTE},
    "crop_type": {
        "classes": CROP_TYPE_CLASSES,
        "palette": CROP_TYPE_PALETTE,
    },
}


@DATASETS.register_module()
class OlmoEarthSegDataset(BaseDataset):
    """Manifest-based MMSeg dataset for OLMoEarth downstream segmentation.

    The manifest deliberately stores explicit array paths and metadata instead
    of constructing rslearn datasets during training. Paths in each sample are
    relative to ``data_root`` unless absolute.
    """

    METAINFO = DATASET_METAINFO["pastis"]

    def __init__(
        self,
        data_root: str | Path,
        ann_file: str | Path,
        pipeline: list[dict[str, Any]] | None = None,
        metainfo: dict[str, Any] | None = None,
        dataset_name: str | None = None,
        test_mode: bool = False,
        lazy_init: bool = False,
        serialize_data: bool = True,
        indices: int | list[int] | None = None,
        max_refetch: int = 1000,
    ) -> None:
        self.dataset_name = dataset_name
        metainfo = metainfo or DATASET_METAINFO.get(
            dataset_name,
            self.METAINFO,
        )
        super().__init__(
            ann_file=str(ann_file),
            metainfo=metainfo,
            data_root=str(data_root),
            data_prefix=dict(),
            filter_cfg=None,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline or [],
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
        )

    def _resolve_path(self, value: str | Path | None) -> str | None:
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute():
            return str(path)
        return str(Path(self.data_root) / path)

    def _resolve_paths(self, values: list[str | Path] | None) -> list[str] | None:
        if values is None:
            return None
        return [self._resolve_path(value) for value in values]

    def load_data_list(self) -> list[dict[str, Any]]:
        with open(self.ann_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        samples = payload["samples"] if isinstance(payload, dict) else payload
        if not isinstance(samples, list):
            raise TypeError(
                "OlmoEarth manifest must be a list or {'samples': list}"
            )

        data_list: list[dict[str, Any]] = []
        for sample in samples:
            item = dict(sample)
            for key in (
                "embedding_path",
                "seg_map_path",
                "valid_mask_path",
            ):
                if key in item:
                    item[key] = self._resolve_path(item[key])
            if "img_paths" in item:
                item["img_paths"] = self._resolve_paths(item["img_paths"])
            item.setdefault("dataset_name", self.dataset_name)
            data_list.append(item)
        return data_list
