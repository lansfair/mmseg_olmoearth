from __future__ import annotations

from typing import Any

import numpy as np
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.structures import PixelData
from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample


@TRANSFORMS.register_module()
class PackOlmoEarthSegInputs(BaseTransform):
    """Pack OLMoEarth arrays for MMSeg while keeping temporal metadata."""

    def __init__(
        self,
        meta_keys=(
            "img_paths",
            "seg_map_path",
            "valid_mask_path",
            "ori_shape",
            "img_shape",
            "dataset_name",
            "sample_id",
            "olmoearth_modality",
            "olmoearth_num_timesteps",
            "olmoearth_band_names",
            "present_bands",
            "timestamps",
            "olmoearth_rgb_adapter",
            "olmoearth_raw_img",
            "olmoearth_raw_band_names",
        ),
    ) -> None:
        self.meta_keys = meta_keys

    @staticmethod
    def _image_to_tensor(array: np.ndarray):
        if array.ndim < 3:
            array = np.expand_dims(array, -1)
        chw = np.ascontiguousarray(array.transpose(2, 0, 1))
        return to_tensor(chw).contiguous()

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        packed: dict[str, Any] = {
            "inputs": self._image_to_tensor(results["img"])
        }

        sample = SegDataSample()
        gt_seg = results["gt_seg_map"]
        if gt_seg.ndim == 2:
            gt_seg = gt_seg[None, ...]
        sample.gt_sem_seg = PixelData(data=to_tensor(gt_seg.astype(np.int64)))

        if "gt_valid_mask" in results:
            valid = results["gt_valid_mask"]
            if valid.ndim == 2:
                valid = valid[None, ...]
            sample.set_data(
                {
                    "gt_valid_mask": PixelData(
                        data=to_tensor(valid.astype(np.float32))
                    )
                }
            )

        metainfo = {
            key: results[key] for key in self.meta_keys if key in results
        }
        sample.set_metainfo(metainfo)
        packed["data_samples"] = sample
        return packed
