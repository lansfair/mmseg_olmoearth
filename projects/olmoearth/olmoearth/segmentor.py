from __future__ import annotations

from typing import List

from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.registry import MODELS
from mmseg.utils import OptSampleList, SampleList
from torch import Tensor


@MODELS.register_module()
class OlmoEarthEncoderDecoder(EncoderDecoder):
    """EncoderDecoder with OLMoEarth temporal metadata passing."""

    def _set_backbone_metainfo(
        self,
        data_samples: OptSampleList = None,
    ) -> None:
        if not hasattr(self.backbone, "set_batch_metainfo"):
            return
        if data_samples is None:
            self.backbone.set_batch_metainfo(None)
        else:
            self.backbone.set_batch_metainfo(
                [data_sample.metainfo for data_sample in data_samples]
            )

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        self._set_backbone_metainfo(data_samples)
        return super().loss(inputs, data_samples)

    def predict(
        self,
        inputs: Tensor,
        data_samples: OptSampleList = None,
    ) -> SampleList:
        self._set_backbone_metainfo(data_samples)
        return super().predict(inputs, data_samples)

    def _forward(
        self,
        inputs: Tensor,
        data_samples: OptSampleList = None,
    ) -> Tensor:
        self._set_backbone_metainfo(data_samples)
        return super()._forward(inputs, data_samples)

    def encode_decode(
        self,
        inputs: Tensor,
        batch_img_metas: List[dict],
    ) -> Tensor:
        if hasattr(self.backbone, "set_batch_metainfo"):
            self.backbone.set_batch_metainfo(batch_img_metas)
        return super().encode_decode(inputs, batch_img_metas)
