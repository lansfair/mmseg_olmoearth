from __future__ import annotations

import os
from typing import Any

from mmengine.dataset import BaseDataset
from mmseg.registry import DATASETS

from ..utils import CROP_TYPE_CLASSES, CROP_TYPE_PALETTE


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


def _normalize_task_name(name: str) -> str:
    return str(name).lower().replace("_", "-").replace(" ", "")


def _set_geobench_root(geobench_root: str | None) -> None:
    if geobench_root is not None:
        # GEO-Bench discovers its local cache through this variable.
        os.environ["GEO_BENCH_DIR"] = geobench_root


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

    METAINFO = dict(classes=CROP_TYPE_CLASSES, palette=CROP_TYPE_PALETTE)

    def __init__(
        self,
        task_name: str = "m-SA-crop-type",
        benchmark_name: str = "segmentation_v1.0",
        split: str = "train",
        partition_name: str = "default",
        band_names: tuple[str, ...] = GEOBENCH_S2_BANDS,
        geobench_format: str = "hdf5",
        geobench_root: str | None = None,
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
                dataset_name="crop_type",
                olmoearth_modality="sentinel2_l2a",
                olmoearth_num_timesteps=1,
                olmoearth_band_names=list(self.band_names),
            )
            for idx in range(len(dataset))
        ]
