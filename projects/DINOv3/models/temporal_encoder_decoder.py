from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.models.segmentors import EncoderDecoder
from mmseg.registry import MODELS


@MODELS.register_module()
class SpectralProjection(BaseModule):
    """Learnable per-pixel projection from 13 Sentinel-2 bands to RGB width."""

    def __init__(
        self,
        in_channels: int = 13,
        out_channels: int = 3,
        with_norm: bool = True,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.proj = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(self.out_channels) if with_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f'SpectralProjection expects (N, {self.in_channels}, H, W), '
                f'got {tuple(x.shape)}.'
            )
        return self.norm(self.proj(x))


@MODELS.register_module()
class TemporalEncoderDecoder(EncoderDecoder):
    """EncoderDecoder for multi-temporal inputs shaped ``B,T,C,H,W``.

    Each time step independently passes through the same spectral projection
    and DINOv3 backbone. Backbone feature maps are fused over time by a fixed
    ``mean`` or ``max`` reduction, then sent to the decode head.
    """

    def __init__(
        self,
        input_projection: dict,
        temporal_fusion: str = 'mean',
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if temporal_fusion not in {'mean', 'max'}:
            raise ValueError(
                f'temporal_fusion must be "mean" or "max", got '
                f'{temporal_fusion!r}.'
            )
        self.input_projection = MODELS.build(input_projection)
        self.temporal_fusion = temporal_fusion

    def _fuse_time(self, feature: torch.Tensor, batch: int, times: int) -> torch.Tensor:
        feature = feature.reshape(batch, times, *feature.shape[1:])
        if self.temporal_fusion == 'mean':
            return feature.mean(dim=1)
        return feature.max(dim=1).values

    def extract_feat(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if inputs.ndim == 4:
            inputs = inputs.unsqueeze(1)
        if inputs.ndim != 5:
            raise ValueError(
                'TemporalEncoderDecoder expects (B,T,C,H,W) or (B,C,H,W), '
                f'got {tuple(inputs.shape)}.'
            )

        batch, times, channels, height, width = inputs.shape
        flattened = inputs.reshape(batch * times, channels, height, width)
        projected = self.input_projection(flattened)
        features = self.backbone(projected)
        if self.with_neck:
            features = self.neck(features)
        if torch.is_tensor(features):
            features = (features,)
        return tuple(self._fuse_time(feat, batch, times) for feat in features)
