import torch
import torch
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import OptSampleList, SampleList, add_prefix

from .segmentors import CopernicusEncoderDecoder


@MODELS.register_module()
class TemporalCopernicusEncoderDecoder(CopernicusEncoderDecoder):
    """Copernicus-FM segmentor for per-timestep temporal aggregation.

    Inputs are packed as B,T*C,H,W, while ``copernicus_meta`` is B,T,4.
    The segmentor restores the temporal dimension, forwards each timestep with
    its own metadata, then averages logits over time.
    """

    def _validate_copernicus_meta(self, data_samples):
        super()._validate_copernicus_meta(data_samples)
        for data_sample in data_samples:
            metainfo = self._get_metainfo(data_sample)
            meta = torch.as_tensor(metainfo[self.meta_key])
            if meta.ndim != 2 or meta.shape[-1] != 4:
                raise ValueError(
                    f'{self.meta_key} must have shape T,4 for '
                    'TemporalCopernicusEncoderDecoder.')

    def _flatten_temporal_inputs(self, inputs, meta):
        batch_size, channels, height, width = inputs.shape
        num_times = meta.shape[1]
        if channels % num_times != 0:
            raise ValueError(
                f'Input has {channels} channels, which is not divisible by '
                f'{num_times} temporal metadata rows.')
        channels_per_time = channels // num_times
        inputs = inputs.reshape(batch_size, num_times, channels_per_time,
                                height, width)
        inputs = inputs.reshape(batch_size * num_times, channels_per_time,
                                height, width)
        meta = meta.reshape(batch_size * num_times, meta.shape[-1])
        return inputs, meta, num_times

    def _extract_temporal_feat(self, inputs, data_samples, required=False):
        meta = self._stack_copernicus_meta(data_samples, inputs,
                                           required=required)
        if meta is None:
            raise KeyError(
                f'Missing {self.meta_key}; '
                'TemporalCopernicusEncoderDecoder requires per-timestep '
                'metadata.')
        if meta.ndim != 3 or meta.shape[-1] != 4:
            raise ValueError(
                f'{self.meta_key} must have shape B,T,4 after batching, '
                f'but got {tuple(meta.shape)}.')
        inputs, meta, num_times = self._flatten_temporal_inputs(inputs, meta)
        x = self.backbone(inputs, meta)
        if self.with_neck:
            x = self.neck(x)
        return x, num_times

    def _aggregate_temporal_logits(self, seg_logits, num_times):
        batch_times, channels, height, width = seg_logits.shape
        if batch_times % num_times != 0:
            raise ValueError(
                f'Logit batch size {batch_times} is not divisible by '
                f'{num_times} timesteps.')
        batch_size = batch_times // num_times
        seg_logits = seg_logits.reshape(batch_size, num_times, channels,
                                        height, width)
        return seg_logits.mean(dim=1)

    def _temporal_head_loss(self, head, inputs, num_times, data_samples):
        seg_logits = head.forward(inputs)
        seg_logits = self._aggregate_temporal_logits(seg_logits, num_times)
        return head.loss_by_feat(seg_logits, data_samples)

    def encode_decode(self, inputs: Tensor, batch_img_metas: list) -> Tensor:
        x, num_times = self._extract_temporal_feat(inputs, batch_img_metas)
        seg_logits = self.decode_head.forward(x)
        seg_logits = self._aggregate_temporal_logits(seg_logits, num_times)
        return self.decode_head.predict_by_feat(seg_logits, batch_img_metas)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        x, num_times = self._extract_temporal_feat(inputs, data_samples,
                                                  required=True)
        losses = dict()
        loss_decode = self._temporal_head_loss(self.decode_head, x,
                                               num_times, data_samples)
        losses.update(add_prefix(loss_decode, 'decode'))
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, torch.nn.ModuleList):
                for idx, aux_head in enumerate(self.auxiliary_head):
                    loss_aux = self._temporal_head_loss(
                        aux_head, x, num_times, data_samples)
                    losses.update(add_prefix(loss_aux, f'aux_{idx}'))
            else:
                loss_aux = self._temporal_head_loss(
                    self.auxiliary_head, x, num_times, data_samples)
                losses.update(add_prefix(loss_aux, 'aux'))
        return losses

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        x, num_times = self._extract_temporal_feat(inputs, data_samples)
        seg_logits = self.decode_head.forward(x)
        return self._aggregate_temporal_logits(seg_logits, num_times)
