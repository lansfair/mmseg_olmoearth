from __future__ import annotations

from typing import Any

from mmengine.model import BaseModel
from mmseg.registry import MODELS


@MODELS.register_module()
class GeoFMEmbeddingModel(BaseModel):
    """Minimal model container used by the standalone embedding exporter."""

    def __init__(self, backbone: dict, data_preprocessor: dict | None = None) -> None:
        super().__init__(data_preprocessor=data_preprocessor)
        self.backbone = MODELS.build(backbone)

    def forward(
        self,
        inputs: Any,
        data_samples=None,
        mode: str = "tensor",
    ):
        if mode != "tensor":
            raise RuntimeError(
                "GeoFMEmbeddingModel is extraction-only; use mode='tensor'."
            )
        metadata = None
        if data_samples is not None:
            metadata = [sample.metainfo for sample in data_samples]
        self.backbone.set_batch_metainfo(metadata)
        return self.backbone(inputs)
