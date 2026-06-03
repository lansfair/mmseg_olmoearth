from __future__ import annotations

import os
from typing import Any

from mmengine.dataset import BaseDataset
from mmseg.registry import DATASETS

from ..utils import (
    CASHEW_PLANT_CLASSES,
    CASHEW_PLANT_PALETTE,
    CROP_TYPE_CLASSES,
    CROP_TYPE_PALETTE,
)


GEOBENCH_S2_BANDS = (
    "02",
    "03",
    "04",
    "08",
    "05",
    "06",
    "07",
    "08A",
    "11",
    "12",
    "01",
    "09",
)

GEOBENCH_METAINFO = {
    "m-SA-crop-type": dict(
        classes=CROP_TYPE_CLASSES,
        palette=CROP_TYPE_PALETTE,
        dataset_name="crop_type",
    ),
    "m-cashew-plant": dict(
        classes=CASHEW_PLANT_CLASSES,
        palette=CASHEW_PLANT_PALETTE,
        dataset_name="cashew_plant",
    ),
}


def _normalize_task_name(name: str) -> str:
    return str(name).lower().replace("_", "-").replace(" ", "")


def _set_geobench_root(geobench_root: str | None) -> None:
    if geobench_root is not None:
        # GEO-Bench discovers its local cache through this variable.
        os.environ["GEO_BENCH_DIR"] = geobench_root


def _get_task_defaults(task_name: str) -> dict[str, Any]:
    normalized = _normalize_task_name(task_name)
    for name, defaults in GEOBENCH_METAINFO.items():
        if _normalize_task_name(name) == normalized:
            return defaults.copy()
    return {}


def _make_generic_palette(num_classes: int) -> list[list[int]]:
    palette = []
    for idx in range(num_classes):
        palette.append(
            [
                (idx * 37) % 256,
                (idx * 67) % 256,
                (idx * 97) % 256,
            ]
        )
    return palette


def get_geobench_task(
    task_name: str,
    benchmark_name: str,
    geobench_root: str | None = None,
):
    _set_geobench_root(geobench_root)
    try:
        import geobench

        task_iterator = geobench.task_iterator
    except Exception:
        from geobench.task import task_iterator

    target = _normalize_task_name(task_name)
    candidates = []
    for task in task_iterator(benchmark_name=benchmark_name):
        candidates.append(task.dataset_name)
        if _normalize_task_name(task.dataset_name) == target:
            return task

    raise RuntimeError(
        f"Cannot find GEO-Bench task {task_name!r} in benchmark "
        f"{benchmark_name!r}. Available tasks: {candidates}. "
        "Check geobench_root and confirm the GEO-Bench data is downloaded."
    )


@DATASETS.register_module()
class GeoBenchS2SegDataset(BaseDataset):
    """GEO-Bench Sentinel-2 segmentation dataset for OLMoEarth probes.

    This dataset stores only sample indices in MMSeg. The actual GEO-Bench
    sample is loaded by ``LoadGeoBenchS2OfficialNorm`` so worker processes can
    cache the underlying GEO-Bench dataset object locally.
    """

    METAINFO = {}

    def __init__(
        self,
        task_name: str = "m-SA-crop-type",
        benchmark_name: str = "segmentation_v1.0",
        split: str = "train",
        partition_name: str = "default",
        band_names: tuple[str, ...] = GEOBENCH_S2_BANDS,
        geobench_format: str = "hdf5",
        geobench_root: str | None = None,
        dataset_name: str | None = None,
        num_classes: int | None = None,
        classes: tuple[str, ...] | list[str] | None = None,
        palette: list[list[int]] | tuple[tuple[int, int, int], ...] | None = None,
        pipeline: list[dict[str, Any]] | None = None,
        metainfo: dict[str, Any] | None = None,
        test_mode: bool = False,
        lazy_init: bool = False,
        **kwargs,
    ) -> None:
        self.task_name = task_name
        self.benchmark_name = benchmark_name
        self.split = split
        self.partition_name = partition_name
        self.band_names = tuple(band_names)
        self.geobench_format = geobench_format
        self.geobench_root = geobench_root
        task_defaults = _get_task_defaults(task_name)
        self.dataset_name = (
            dataset_name
            or task_defaults.get("dataset_name")
            or _normalize_task_name(task_name)
        )

        if metainfo is None:
            resolved_classes = classes or task_defaults.get("classes")
            resolved_palette = palette or task_defaults.get("palette")
            if resolved_classes is None and num_classes is not None:
                resolved_classes = tuple(f"class_{idx}" for idx in range(num_classes))
            if resolved_palette is None and resolved_classes is not None:
                resolved_palette = _make_generic_palette(len(resolved_classes))
            metainfo = {}
            if resolved_classes is not None:
                metainfo["classes"] = tuple(resolved_classes)
            if resolved_palette is not None:
                metainfo["palette"] = [list(color) for color in resolved_palette]

        super().__init__(
            ann_file="",
            metainfo=metainfo,
            data_root="",
            data_prefix={},
            pipeline=pipeline or [],
            test_mode=test_mode,
            lazy_init=lazy_init,
            serialize_data=False,
            **kwargs,
        )

    def load_data_list(self) -> list[dict[str, Any]]:
        task = get_geobench_task(
            self.task_name,
            self.benchmark_name,
            self.geobench_root,
        )
        dataset = task.get_dataset(
            split=self.split,
            partition_name=self.partition_name,
            band_names=self.band_names,
            format=self.geobench_format,
        )
        return [
            dict(
                sample_idx=idx,
                task_name=self.task_name,
                benchmark_name=self.benchmark_name,
                split=self.split,
                partition_name=self.partition_name,
                band_names=list(self.band_names),
                geobench_format=self.geobench_format,
                geobench_root=self.geobench_root,
                dataset_name=self.dataset_name,
                olmoearth_modality="sentinel2_l2a",
                olmoearth_num_timesteps=1,
                olmoearth_band_names=list(self.band_names),
            )
            for idx in range(len(dataset))
        ]
