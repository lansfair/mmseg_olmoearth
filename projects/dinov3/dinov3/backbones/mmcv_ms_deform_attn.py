from __future__ import annotations

import inspect

import torch
import torch.nn as nn
from mmcv.ops import MultiScaleDeformableAttention
from torch import Tensor


class MSDeformAttn(nn.Module):
    """DINOv3-compatible wrapper around MMCV deformable attention."""

    def __init__(
        self,
        d_model: int = 256,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
        ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio
        if "value_proj_ratio" in inspect.signature(
            MultiScaleDeformableAttention.__init__
        ).parameters:
            self.attn = MultiScaleDeformableAttention(
                embed_dims=d_model,
                num_levels=n_levels,
                num_heads=n_heads,
                num_points=n_points,
                dropout=0.0,
                batch_first=True,
                value_proj_ratio=ratio,
            )
        else:
            raise RuntimeError(
                "mmcv.ops.MultiScaleDeformableAttention must support "
                "value_proj_ratio. Please use mmcv>=2.0.0rc4."
            )

    def init_weights(self) -> None:
        self.attn.init_weights()

    def _reset_parameters(self) -> None:
        self.init_weights()

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        input_flatten: Tensor,
        input_spatial_shapes: Tensor,
        input_level_start_index: Tensor,
        input_padding_mask: Tensor | None = None,
    ) -> Tensor:
        return self.attn(
            query=query,
            value=input_flatten,
            identity=torch.zeros_like(query),
            query_pos=None,
            key_padding_mask=input_padding_mask,
            reference_points=reference_points,
            spatial_shapes=input_spatial_shapes,
            level_start_index=input_level_start_index,
        )
