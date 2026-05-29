from __future__ import annotations

from typing import Any

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS
from scipy.ndimage import zoom

from ..datasets.geobench import get_geobench_task


OFFICIAL_EVAL_S2_BAND_NAMES = [
    "01 - Coastal aerosol",
    "02 - Blue",
    "03 - Green",
    "04 - Red",
    "05 - Vegetation Red Edge",
    "06 - Vegetation Red Edge",
    "07 - Vegetation Red Edge",
    "08 - NIR",
    "08A - Vegetation Red Edge",
    "09 - Water vapour",
    "10 - SWIR - Cirrus",
    "11 - SWIR",
    "12 - SWIR",
]

OFFICIAL_OLMOEARTH_S2_L2A_BAND_NAMES = [
    "02 - Blue",
    "03 - Green",
    "04 - Red",
    "08 - NIR",
    "05 - Vegetation Red Edge",
    "06 - Vegetation Red Edge",
    "07 - Vegetation Red Edge",
    "08A - Vegetation Red Edge",
    "11 - SWIR",
    "12 - SWIR",
    "01 - Coastal aerosol",
    "09 - Water vapour",
]

M_SA_CROP_TYPE_IMPUTES = [("11 - SWIR", "10 - SWIR - Cirrus")]


def _stats_to_float(stats: Any, key: str) -> float:
    if isinstance(stats, dict):
        return float(stats[key])
    return float(getattr(stats, key))


def _impute_band_stats(
    band_stats: dict,
    imputes: list[tuple[str, str]],
    all_bands: list[str],
) -> dict:
    new_band_stats = {}
    for band_name in all_bands:
        if band_name in band_stats:
            new_band_stats[band_name] = band_stats[band_name]
            continue
        for source, target in imputes:
            if target == band_name:
                if source not in band_stats:
                    raise KeyError(
                        f"Cannot impute stats for {target}: "
                        f"source {source} not found."
                    )
                new_band_stats[band_name] = band_stats[source]
                break
        else:
            raise KeyError(
                f"Band stats for {band_name} not found and no impute rule "
                "matched."
            )
    return new_band_stats


def _norm_no_clip_2_std(
    image: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    means = means.reshape(1, 1, -1).astype(np.float32)
    stds = stds.reshape(1, 1, -1).astype(np.float32)
    lower = means - 2.0 * stds
    upper = means + 2.0 * stds
    return (image.astype(np.float32) - lower) / (upper - lower)


def _collect_s2_bands(sample, all_bands: list[str]) -> np.ndarray:
    band_map = {band.band_info.name: band.data for band in sample.bands}
    image_list = []
    for band_name in all_bands:
        if band_name in band_map:
            image_list.append(band_map[band_name])
            continue
        if band_name == "10 - SWIR - Cirrus":
            source = "11 - SWIR"
            if source not in band_map:
                raise KeyError(
                    f"Cannot impute {band_name}: source band {source} "
                    "not found."
                )
            image_list.append(band_map[source])
            continue
        raise KeyError(
            f"Cannot find required band {band_name}. "
            f"Available bands: {list(band_map.keys())}"
        )
    return np.stack(image_list, axis=-1)


def _resize_2d_if_needed(
    array: np.ndarray,
    target_hw: tuple[int, int],
    order: int,
) -> np.ndarray:
    if array.shape[:2] == target_hw:
        return array
    zoom_factor = (target_hw[0] / array.shape[0], target_hw[1] / array.shape[1])
    return zoom(array, zoom=zoom_factor, order=order)


@TRANSFORMS.register_module()
class LoadGeoBenchS2OfficialNorm(BaseTransform):
    """Load GEO-Bench S2 samples with OLMoEarth official normalization.

    For m-SA-crop-type this reads the official 13-band Sentinel-2 sample,
    imputes B10 from B11, applies ``NORM_NO_CLIP_2_STD`` using task statistics,
    and selects the OLMoEarth Sentinel-2 L2A 12-band order.
    """

    def __init__(
        self,
        num_classes: int = 10,
        ignore_index: int = 255,
        invalid_label_to_ignore: bool = True,
        label_resample_order: int = 0,
        default_timestamp: tuple[int, int, int] = (15, 4, 2024),
        keep_raw_input: bool = False,
    ) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.invalid_label_to_ignore = bool(invalid_label_to_ignore)
        self.label_resample_order = int(label_resample_order)
        self.default_timestamp = tuple(int(x) for x in default_timestamp)
        self.keep_raw_input = bool(keep_raw_input)
        self._dataset = None
        self._task = None
        self._dataset_key = None
        self._means_13 = None
        self._stds_13 = None
        self._olmo_indices = None

    def _get_dataset_and_task(self, results: dict):
        key = (
            results["task_name"],
            results["benchmark_name"],
            results["split"],
            results["partition_name"],
            results["geobench_format"],
            results.get("geobench_root"),
        )
        if self._dataset is not None and self._dataset_key == key:
            return self._dataset, self._task

        task = get_geobench_task(
            results["task_name"],
            results["benchmark_name"],
            results.get("geobench_root"),
        )
        dataset = task.get_dataset(
            split=results["split"],
            partition_name=results["partition_name"],
            format=results["geobench_format"],
        )
        band_stats = _impute_band_stats(
            task.band_stats,
            M_SA_CROP_TYPE_IMPUTES,
            OFFICIAL_EVAL_S2_BAND_NAMES,
        )
        self._means_13 = np.asarray(
            [
                _stats_to_float(band_stats[band_name], "mean")
                for band_name in OFFICIAL_EVAL_S2_BAND_NAMES
            ],
            dtype=np.float32,
        )
        self._stds_13 = np.asarray(
            [
                _stats_to_float(band_stats[band_name], "std")
                for band_name in OFFICIAL_EVAL_S2_BAND_NAMES
            ],
            dtype=np.float32,
        )
        name_to_idx = {
            name: idx for idx, name in enumerate(OFFICIAL_EVAL_S2_BAND_NAMES)
        }
        self._olmo_indices = [
            name_to_idx[name]
            for name in OFFICIAL_OLMOEARTH_S2_L2A_BAND_NAMES
        ]
        self._dataset = dataset
        self._task = task
        self._dataset_key = key
        return self._dataset, self._task

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        dataset, _ = self._get_dataset_and_task(results)
        sample = dataset[results["sample_idx"]]
        image_13 = _collect_s2_bands(
            sample,
            OFFICIAL_EVAL_S2_BAND_NAMES,
        ).astype(np.float32)
        raw_image = image_13.astype(np.float32)
        image_13 = _norm_no_clip_2_std(
            image_13,
            self._means_13,
            self._stds_13,
        )
        image = image_13[:, :, self._olmo_indices].astype(np.float32)

        label = np.asarray(sample.label.data)
        if label.ndim == 3:
            if label.shape[-1] == 1:
                label = label[..., 0]
            elif label.shape[0] == 1:
                label = label[0]
            else:
                raise RuntimeError(
                    f"Unexpected label shape for sample "
                    f"{results['sample_idx']}: {label.shape}"
                )
        label = _resize_2d_if_needed(
            label,
            image.shape[:2],
            self.label_resample_order,
        )
        if np.issubdtype(label.dtype, np.floating):
            invalid = ~np.isfinite(label)
            label = np.rint(label).astype(np.int64)
            label[invalid] = self.ignore_index
        else:
            label = label.astype(np.int64)
        if self.invalid_label_to_ignore:
            invalid = (label < 0) | (label >= self.num_classes)
            label[invalid] = self.ignore_index

        results["img"] = np.ascontiguousarray(image)
        if self.keep_raw_input:
            results["olmoearth_raw_img"] = np.ascontiguousarray(raw_image)
            results["olmoearth_raw_band_names"] = OFFICIAL_EVAL_S2_BAND_NAMES
        results["gt_seg_map"] = np.ascontiguousarray(label)
        results["img_shape"] = image.shape[:2]
        results["ori_shape"] = image.shape[:2]
        results["seg_fields"] = ["gt_seg_map"]
        results["sample_name"] = getattr(
            sample,
            "sample_name",
            str(results["sample_idx"]),
        )
        results["timestamps"] = np.asarray(
            [self.default_timestamp],
            dtype=np.int64,
        )
        return results
