from __future__ import annotations

from mmengine.model import BaseModule
from mmseg.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class TesseraFeatureBackbone(BaseModule):
    """Backbone wrapper for precomputed dense TESSERA embeddings.

    TESSERA embeddings are already pixel-aligned dense features, so the
    backbone simply validates the channel count and returns a single feature
    map for a segmentation head.
    """

    def __init__(
        self,
        out_channels: int = 128,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.out_channels = out_channels

    def forward(self, inputs: Tensor) -> tuple[Tensor]:
        if inputs.shape[1] != self.out_channels:
            raise ValueError(
                f"Expected {self.out_channels} TESSERA embedding channels, "
                f"got {inputs.shape[1]}"
            )
        return (inputs,)
