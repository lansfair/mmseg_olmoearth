from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmengine.dataset import BaseDataset
from mmseg.registry import DATASETS

CROP_TYPE_CLASSES = (
    "Background",
    "Lucerne/Medics",
    "Planted pastures (perennial)",
    "Fallow",
    "Wine grapes",
    "Weeds",
    "Small grain grazing",
    "Wheat",
    "Canola",
    "Rooibos",
)
CROP_TYPE_PALETTE = [
    [0, 0, 0], [166, 206, 227], [31, 120, 180], [178, 223, 138],
    [51, 160, 44], [251, 154, 153], [227, 26, 28], [253, 191, 111],
    [255, 127, 0], [202, 178, 214],
]
DATASET_METAINFO = {
    "crop_type": {"classes": CROP_TYPE_CLASSES, "palette": CROP_TYPE_PALETTE},
}


@DATASETS.register_module()
class TesseraSegDataset(BaseDataset):
    """Manifest-based MMSeg dataset for precomputed TESSERA embeddings.

    Each manifest sample should provide at least ``embedding_path`` and
    ``seg_map_path``. For QAT outputs, provide ``scales_path`` or
    ``scale_path`` so ``LoadTesseraEmbedding`` can dequantize int8 features.
    Paths are relative to ``data_root`` unless absolute.
    """

    METAINFO = DATASET_METAINFO["crop_type"]

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

    def load_data_list(self) -> list[dict[str, Any]]:
        with open(self.ann_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        samples = payload["samples"] if isinstance(payload, dict) else payload
        if not isinstance(samples, list):
            raise TypeError(
                "TESSERA manifest must be a list or {'samples': list}"
            )

        data_list: list[dict[str, Any]] = []
        for sample in samples:
            item = dict(sample)
            for key in (
                "tile_path",
                "embedding_path",
                "scales_path",
                "scale_path",
                "bands_path",
                "masks_path",
                "doys_path",
                "sar_ascending_path",
                "sar_ascending_doy_path",
                "sar_descending_path",
                "sar_descending_doy_path",
                "seg_map_path",
                "valid_mask_path",
            ):
                if key in item:
                    item[key] = self._resolve_path(item[key])
            item.setdefault("dataset_name", self.dataset_name)
            data_list.append(item)
        return data_list
