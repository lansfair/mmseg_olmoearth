from __future__ import annotations

import torch.nn as nn
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class DINOv3PASTISUpHead(BaseDecodeHead):
    """Course-style upsampling decoder for DINOv3 PASTIS features."""

    def __init__(
        self,
        input_size: int = 8,
        output_size: int = 128,
        **kwargs,
    ) -> None:
        kwargs.setdefault("dropout_ratio", 0.1)
        super().__init__(input_transform=None, **kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.input_proj = (
            nn.Identity()
            if self.in_channels == self.channels
            else nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        )

        layers = []
        cur = input_size
        while cur < output_size:
            layers.append(
                nn.Sequential(
                    nn.Upsample(
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=self.align_corners,
                    ),
                    nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True),
                )
            )
            cur *= 2
        self.ups = nn.ModuleList(layers)
        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

    def forward(self, inputs: Tensor | tuple[Tensor, ...] | list[Tensor]) -> Tensor:
        x = self._transform_inputs(inputs)
        x = self.input_proj(x)
        for up in self.ups:
            x = up(x)
        return self.cls_seg(x)
