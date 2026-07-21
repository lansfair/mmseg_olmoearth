from __future__ import annotations

from typing import Any

import numpy as np
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.structures import PixelData
from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample


@TRANSFORMS.register_module()
class PackTesseraSegInputs(BaseTransform):
    """Pack TESSERA arrays into MMSeg inputs and data samples."""

    def __init__(
        self,
        meta_keys=(
            "tile_path",
            "embedding_path",
            "seg_map_path",
            "valid_mask_path",
            "ori_shape",
            "img_shape",
            "dataset_name",
            "sample_id",
        ),
    ) -> None:
        self.meta_keys = meta_keys

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        image = results["img"]
        if image.ndim < 3:
            image = image[..., None]
        inputs = to_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        label = results["gt_seg_map"]
        if label.ndim == 2:
            label = label[None, ...]
        sample = SegDataSample()
        sample.gt_sem_seg = PixelData(data=to_tensor(label.astype(np.int64)))
        if "gt_valid_mask" in results:
            valid = results["gt_valid_mask"]
            if valid.ndim == 2:
                valid = valid[None, ...]
            sample.set_data({
                "gt_valid_mask": PixelData(
                    data=to_tensor(valid.astype(np.float32))
                )
            })
        sample.set_metainfo({key: results[key] for key in self.meta_keys if key in results})
        return {"inputs": inputs.contiguous(), "data_samples": sample}
