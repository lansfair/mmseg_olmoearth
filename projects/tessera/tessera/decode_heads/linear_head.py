from __future__ import annotations

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from torch import Tensor
from torch.nn import functional as F


@MODELS.register_module()
class TesseraLinearHead(BaseDecodeHead):
    """Single-layer segmentation probe for dense TESSERA features."""

    def __init__(self, scale_factor: int = 1, **kwargs) -> None:
        kwargs.setdefault("dropout_ratio", 0)
        super().__init__(input_transform=None, **kwargs)
        self.scale_factor = int(scale_factor)

    def forward(self, inputs: Tensor | tuple[Tensor, ...]) -> Tensor:
        x = self._transform_inputs(inputs)
        if self.scale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode="bilinear",
                align_corners=self.align_corners,
            )
        return self.cls_seg(x)
