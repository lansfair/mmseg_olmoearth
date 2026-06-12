import torch
from torch import Tensor

from mmseg.models.segmentors import EncoderDecoder
from mmseg.registry import MODELS
from mmseg.utils import OptSampleList, SampleList


@MODELS.register_module()
class CopernicusEncoderDecoder(EncoderDecoder):
    """EncoderDecoder that forwards Copernicus-FM metadata to the backbone."""

    meta_key = 'copernicus_meta'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.test_cfg is not None and self.test_cfg.get(
                'mode', 'whole') != 'whole':
            raise ValueError('CopernicusEncoderDecoder currently supports '
                             'only whole inference.')

    def _get_metainfo(self, data_sample):
        return data_sample.metainfo if hasattr(data_sample,
                                               'metainfo') else data_sample

    def _validate_copernicus_meta(self, data_samples):
        for data_sample in data_samples:
            metainfo = self._get_metainfo(data_sample)
            if self.meta_key not in metainfo:
                raise KeyError(
                    f'Missing {self.meta_key} in data sample metainfo. '
                    'Please add AddCopernicusMeta and include it in '
                    'PackSegInputs.meta_keys.')

    def _stack_copernicus_meta(self, data_samples, inputs, required=False):
        if data_samples is None:
            return None
        metas = []
        for data_sample in data_samples:
            metainfo = self._get_metainfo(data_sample)
            meta = metainfo.get(self.meta_key)
            if meta is None:
                if required:
                    raise KeyError(
                        f'Missing {self.meta_key} in data sample metainfo. '
                        'Please add AddCopernicusMeta and include it in '
                        'PackSegInputs.meta_keys.')
                return None
            metas.append(torch.as_tensor(meta, dtype=inputs.dtype))
        return torch.stack(metas, dim=0).to(inputs.device)

    def extract_feat(self,
                     inputs: Tensor,
                     data_samples: OptSampleList = None,
                     required_meta=False) -> list:
        meta = self._stack_copernicus_meta(data_samples, inputs,
                                           required=required_meta)
        x = self.backbone(inputs, meta)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor, batch_img_metas: list) -> Tensor:
        x = self.extract_feat(inputs, batch_img_metas)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)
        return seg_logits

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        x = self.extract_feat(inputs, data_samples, required_meta=True)
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)
        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        if data_samples is not None:
            self._validate_copernicus_meta(data_samples)
        return super().predict(inputs, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        x = self.extract_feat(inputs, data_samples)
        return self.decode_head.forward(x)
