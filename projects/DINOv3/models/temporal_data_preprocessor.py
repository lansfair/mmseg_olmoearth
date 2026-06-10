from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from mmengine.model import BaseDataPreprocessor
from mmseg.registry import MODELS


@MODELS.register_module()
class TemporalSegDataPreProcessor(BaseDataPreprocessor):
    """Stack/pad temporal samples while preserving ``T,C,H,W`` per sample.

    Spectral normalization is intentionally performed in the dataset pipeline,
    because its statistics are loaded from the user-supplied JSON file.
    """

    def __init__(
        self,
        pad_val: float = 0.0,
        seg_pad_val: int = 255,
        size_divisor: int = 16,
        non_blocking: bool = False,
    ) -> None:
        super().__init__(non_blocking=non_blocking)
        self.pad_val = float(pad_val)
        self.seg_pad_val = int(seg_pad_val)
        self.size_divisor = int(size_divisor)

    def _target_size(self, inputs: List[torch.Tensor]) -> tuple[int, int]:
        height = max(int(item.shape[-2]) for item in inputs)
        width = max(int(item.shape[-1]) for item in inputs)
        if self.size_divisor > 1:
            height = math.ceil(height / self.size_divisor) * self.size_divisor
            width = math.ceil(width / self.size_divisor) * self.size_divisor
        return height, width

    def forward(self, data: dict, training: bool = False) -> dict:
        data = self.cast_data(data)
        inputs = data['inputs']
        data_samples: Optional[list] = data.get('data_samples')

        if isinstance(inputs, torch.Tensor):
            if inputs.ndim == 5:
                input_list = list(inputs)
            elif inputs.ndim == 4:
                input_list = [inputs]
            else:
                raise ValueError(f'Unexpected batched input shape: {tuple(inputs.shape)}')
        else:
            input_list = list(inputs)

        if not input_list or any(item.ndim != 4 for item in input_list):
            shapes = [tuple(item.shape) for item in input_list]
            raise ValueError(f'Each temporal sample must be (T,C,H,W), got {shapes}.')

        target_h, target_w = self._target_size(input_list)
        padded_inputs = []
        for item in input_list:
            pad_h = target_h - int(item.shape[-2])
            pad_w = target_w - int(item.shape[-1])
            padded_inputs.append(F.pad(item, (0, pad_w, 0, pad_h), value=self.pad_val))
        batch_inputs = torch.stack(padded_inputs, dim=0)

        if data_samples is not None:
            for sample in data_samples:
                if hasattr(sample, 'gt_sem_seg'):
                    seg = sample.gt_sem_seg.data
                    pad_h = target_h - int(seg.shape[-2])
                    pad_w = target_w - int(seg.shape[-1])
                    sample.gt_sem_seg.data = F.pad(
                        seg, (0, pad_w, 0, pad_h), value=self.seg_pad_val
                    )
                sample.set_metainfo({
                    'batch_input_shape': (target_h, target_w),
                    'pad_shape': (target_h, target_w),
                    'padding_size': [0, 0, 0, 0],
                })

        return dict(inputs=batch_inputs, data_samples=data_samples)
