from __future__ import annotations

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module()
class LinearProbeHead(BaseDecodeHead):
    """A true spatial linear probe: one 1x1 classifier on one feature map."""

    def __init__(self, in_channels: int, num_classes: int, **kwargs) -> None:
        kwargs.setdefault('input_transform', None)
        kwargs.setdefault('dropout_ratio', 0.0)
        super().__init__(
            in_channels=in_channels,
            channels=in_channels,
            num_classes=num_classes,
            **kwargs,
        )

    def forward(self, inputs):
        feature = self._transform_inputs(inputs)
        return self.cls_seg(feature)
