"""Datasets used by the DOFAv2 MMSegmentation recipes."""

from __future__ import annotations

from typing import Any

from mmengine.dataset import BaseDataset
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS

CASHEW_CLASSES = tuple(f'class_{index}' for index in range(7))
CASHEW_PALETTE = (
    (255, 255, 255),
    (255, 0, 0),
    (255, 255, 0),
    (0, 0, 255),
    (159, 129, 183),
    (0, 255, 0),
    (255, 195, 128),
)

DOFA_S2_9_BANDS = (
    '04 - Red',
    '03 - Green',
    '02 - Blue',
    '05 - Vegetation Red Edge',
    '06 - Vegetation Red Edge',
    '07 - Vegetation Red Edge',
    '08 - NIR',
    '11 - SWIR',
    '12 - SWIR',
)


def build_geobench_dataset(dataset_root: str,
                           split: str,
                           partition_name: str,
                           band_names: tuple[str, ...] | list[str],
                           geobench_format: str):
    """Build GEO-Bench directly from an absolute dataset directory."""
    from geobench.dataset import GeobenchDataset

    return GeobenchDataset(
        dataset_dir=dataset_root,
        split=split,
        partition_name=partition_name,
        band_names=band_names,
        format=geobench_format,
    )


@DATASETS.register_module()
class CashewPlantSegDataset(BaseDataset):
    """GEO-Bench m-cashew-plant dataset.

    Only serializable sample metadata is kept in ``data_list``. Each worker's
    loading transform owns its GEO-Bench dataset cache, avoiding the
    split-overwriting module global used by the original migration.
    """

    METAINFO = {
        'classes': CASHEW_CLASSES,
        'palette': CASHEW_PALETTE,
    }

    def __init__(
        self,
        dataset_root: str,
        split: str = 'train',
        partition_name: str = 'default',
        band_names: tuple[str, ...] | list[str] = DOFA_S2_9_BANDS,
        geobench_format: str = 'hdf5',
        pipeline: list[dict[str, Any]] | None = None,
        test_mode: bool = False,
        lazy_init: bool = False,
        **kwargs,
    ):
        self.dataset_root = dataset_root
        self.split = split
        self.partition_name = partition_name
        self.band_names = tuple(band_names)
        self.geobench_format = geobench_format
        super().__init__(
            ann_file='',
            data_root='',
            data_prefix={},
            pipeline=pipeline or [],
            test_mode=test_mode,
            lazy_init=lazy_init,
            serialize_data=False,
            **kwargs,
        )

    def load_data_list(self) -> list[dict[str, Any]]:
        dataset = build_geobench_dataset(
            dataset_root=self.dataset_root,
            split=self.split,
            partition_name=self.partition_name,
            band_names=self.band_names,
            geobench_format=self.geobench_format,
        )
        return [
            {
                'sample_idx': index,
                'dataset_root': self.dataset_root,
                'split': self.split,
                'partition_name': self.partition_name,
                'band_names': list(self.band_names),
                'geobench_format': self.geobench_format,
            } for index in range(len(dataset))
        ]


@DATASETS.register_module()
class SVDTSegDataset(BaseSegDataset):
    METAINFO = {
        'classes': ('background', 'cropland'),
        'palette': ((0, 0, 0), (128, 0, 0)),
    }


@DATASETS.register_module()
class NingBo2MSegDataset(BaseSegDataset):
    METAINFO = {
        'classes': (
            'background',
            'grassland',
            'bareland',
            'road',
            'forest',
            'river',
            'cropland',
            'residential',
        ),
        'palette': (
            (0, 0, 0),
            (128, 0, 0),
            (0, 128, 0),
            (128, 128, 0),
            (0, 0, 128),
            (128, 0, 128),
            (0, 128, 128),
            (128, 128, 128),
        ),
    }
