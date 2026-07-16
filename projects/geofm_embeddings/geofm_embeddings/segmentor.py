from __future__ import annotations

from typing import Any

from mmseg.models.segmentors import EncoderDecoder
from mmseg.registry import MODELS
from mmseg.utils import OptSampleList, SampleList


@MODELS.register_module()
class GeoFMEncoderDecoder(EncoderDecoder):
    """EncoderDecoder that forwards sample metadata to GeoFM adapters."""

    def _set_backbone_metainfo(self, data_samples: OptSampleList = None) -> None:
        if not hasattr(self.backbone, "set_batch_metainfo"):
            return
        metadata = None
        if data_samples is not None:
            metadata = [sample.metainfo for sample in data_samples]
        self.backbone.set_batch_metainfo(metadata)

    def loss(self, inputs: Any, data_samples: SampleList) -> dict:
        self._set_backbone_metainfo(data_samples)
        return super().loss(inputs, data_samples)

    def predict(
        self,
        inputs: Any,
        data_samples: OptSampleList = None,
    ) -> SampleList:
        self._set_backbone_metainfo(data_samples)
        return super().predict(inputs, data_samples)

    def _forward(self, inputs: Any, data_samples: OptSampleList = None):
        self._set_backbone_metainfo(data_samples)
        return super()._forward(inputs, data_samples)

    def encode_decode(self, inputs: Any, batch_img_metas: list[dict]):
        if hasattr(self.backbone, "set_batch_metainfo"):
            self.backbone.set_batch_metainfo(batch_img_metas)
        return super().encode_decode(inputs, batch_img_metas)
