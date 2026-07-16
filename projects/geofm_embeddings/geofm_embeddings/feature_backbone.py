from __future__ import annotations

from mmengine.model import BaseModule
from mmseg.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class PrecomputedEmbeddingBackbone(BaseModule):
    """Validate and expose precomputed dense embeddings to MMSeg heads."""

    def __init__(self, out_channels: int, init_cfg: dict | None = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.out_channels = out_channels

    def forward(self, inputs: Tensor) -> tuple[Tensor]:
        if inputs.ndim != 4:
            raise ValueError(
                "Dense precomputed embeddings must have shape [B, D, H, W]."
            )
        if inputs.shape[1] != self.out_channels:
            raise ValueError(
                f"Expected D={self.out_channels}, got D={inputs.shape[1]}."
            )
        return (inputs,)
