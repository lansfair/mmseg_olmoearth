from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from mmengine.model import BaseDataPreprocessor
from mmseg.registry import MODELS


@MODELS.register_module(force=True)
class PastisTemporalDataPreprocessor(BaseDataPreprocessor):
    """Data preprocessor for PASTIS temporal tensors.

    Keeps input as B x T x C x H x W and optionally normalizes over C.
    Mean/std should be specified after applying scale_factor in the loader.
    """

    def __init__(
        self,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        pad_size_divisor: int = 1,
        pad_value: float = 0.0,
        seg_pad_value: int = -1,
        non_blocking: bool = False,
    ):
        super().__init__(non_blocking=non_blocking)
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.seg_pad_value = seg_pad_value
        self._enable_normalize = mean is not None and std is not None
        if self._enable_normalize:
            mean = torch.tensor(mean, dtype=torch.float32).view(1, 1, -1, 1, 1)
            std = torch.tensor(std, dtype=torch.float32).view(1, 1, -1, 1, 1)
            self.register_buffer('mean', mean, persistent=False)
            self.register_buffer('std', std, persistent=False)

    def _stack_inputs(self, inputs):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs, dim=0)
        if inputs.dim() == 4:
            inputs = inputs.unsqueeze(1)
        if inputs.dim() != 5:
            raise ValueError(f'Expected inputs with shape B x T x C x H x W, got {tuple(inputs.shape)}')
        return inputs

    def forward(self, data: dict, training: bool = False) -> dict:
        data = self.cast_data(data)
        inputs = self._stack_inputs(data['inputs'])
        data_samples = data.get('data_samples', None)

        if self._enable_normalize:
            if inputs.shape[2] != self.mean.shape[2]:
                raise ValueError(f'Normalization channel mismatch: input C={inputs.shape[2]}, mean/std C={self.mean.shape[2]}')
            inputs = (inputs - self.mean) / self.std.clamp_min(1e-6)

        _, _, _, h, w = inputs.shape
        pad_h = (self.pad_size_divisor - h % self.pad_size_divisor) % self.pad_size_divisor
        pad_w = (self.pad_size_divisor - w % self.pad_size_divisor) % self.pad_size_divisor
        if pad_h > 0 or pad_w > 0:
            inputs = F.pad(inputs, (0, pad_w, 0, pad_h), value=self.pad_value)
            if data_samples is not None:
                for sample in data_samples:
                    if hasattr(sample, 'gt_sem_seg'):
                        gt = sample.gt_sem_seg.data
                        sample.gt_sem_seg.data = F.pad(gt, (0, pad_w, 0, pad_h), value=self.seg_pad_value)

        batch_input_shape = tuple(inputs.shape[-2:])
        if data_samples is not None:
            for sample in data_samples:
                sample.set_metainfo(dict(batch_input_shape=batch_input_shape, pad_shape=batch_input_shape))

        return dict(inputs=inputs, data_samples=data_samples)
