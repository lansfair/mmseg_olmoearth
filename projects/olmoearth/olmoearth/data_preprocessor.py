from __future__ import annotations

import torch.nn.functional as F
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.registry import MODELS


@MODELS.register_module()
class OlmoEarthSegDataPreProcessor(SegDataPreProcessor):
    """SegDataPreProcessor that also pads OLMoEarth valid masks."""

    def _pad_valid_masks(self, inputs, data_samples) -> None:
        if data_samples is None:
            return
        height, width = inputs.shape[-2:]
        for data_sample in data_samples:
            if "gt_valid_mask" not in data_sample:
                continue
            valid_mask = data_sample.gt_valid_mask.data
            pad_h = height - valid_mask.shape[-2]
            pad_w = width - valid_mask.shape[-1]
            if pad_h < 0 or pad_w < 0:
                continue
            if pad_h == 0 and pad_w == 0:
                continue
            data_sample.gt_valid_mask.data = F.pad(
                valid_mask,
                (0, pad_w, 0, pad_h),
                value=0,
            )

    def forward(self, data: dict, training: bool = False) -> dict:
        output = super().forward(data, training=training)
        self._pad_valid_masks(output["inputs"], output.get("data_samples"))
        return output
