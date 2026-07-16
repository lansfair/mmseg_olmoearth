from __future__ import annotations

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from torch import Tensor, nn
from torch.nn import functional as F

from .valid_mask_mixin import GeoFMValidMaskLossMixin


class _GeoFMLinearProbeMixin(GeoFMValidMaskLossMixin):
    def loss_by_feat(
        self,
        seg_logits: Tensor,
        batch_data_samples: list,
    ) -> dict:
        seg_label = self._stack_batch_gt(batch_data_samples).squeeze(1).long()
        if seg_logits.shape[-2:] != seg_label.shape[-2:]:
            seg_logits = resize(
                input=seg_logits,
                size=seg_label.shape[1:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        return self._losses_with_optional_valid_mask(
            seg_logits,
            seg_label,
            batch_data_samples,
        )


@MODELS.register_module()
class GeoFMPatchLinearHead(_GeoFMLinearProbeMixin, BaseDecodeHead):
    """OlmoEarth-style affine patch-to-pixel segmentation probe."""

    def __init__(
        self,
        patch_size: int = 4,
        use_valid_mask: bool = False,
        valid_mask_loss: bool = False,
        **kwargs,
    ) -> None:
        kwargs.setdefault("dropout_ratio", 0)
        super().__init__(input_transform=None, **kwargs)
        if patch_size < 1:
            raise ValueError("patch_size must be at least 1.")
        self.patch_size = patch_size
        self.use_valid_mask = use_valid_mask
        self.valid_mask_loss = valid_mask_loss
        self.conv_seg = nn.Identity()
        self.proj = nn.Linear(
            self.in_channels,
            self.num_classes * patch_size * patch_size,
            bias=True,
        )

    def forward(
        self,
        inputs: Tensor | tuple[Tensor, ...] | list[Tensor],
    ) -> Tensor:
        x = self._transform_inputs(inputs)
        bsz, _, height, width = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        logits = self.proj(x)
        patch = self.patch_size
        logits = logits.view(
            bsz,
            height,
            width,
            self.num_classes,
            patch,
            patch,
        )
        logits = logits.permute(0, 3, 1, 4, 2, 5).contiguous()
        return logits.view(
            bsz,
            self.num_classes,
            height * patch,
            width * patch,
        )


@MODELS.register_module()
class GeoFMLinearHead(_GeoFMLinearProbeMixin, BaseDecodeHead):
    """A 1x1 affine probe with optional bilinear output upsampling."""

    def __init__(
        self,
        scale_factor: int | float = 1,
        use_valid_mask: bool = False,
        valid_mask_loss: bool = False,
        **kwargs,
    ) -> None:
        kwargs.setdefault("dropout_ratio", 0)
        super().__init__(input_transform=None, **kwargs)
        if scale_factor <= 0:
            raise ValueError("scale_factor must be positive.")
        self.scale_factor = scale_factor
        self.use_valid_mask = use_valid_mask
        self.valid_mask_loss = valid_mask_loss

    def forward(
        self,
        inputs: Tensor | tuple[Tensor, ...] | list[Tensor],
    ) -> Tensor:
        x = self._transform_inputs(inputs)
        if self.scale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode="bilinear",
                align_corners=self.align_corners,
            )
        return self.cls_seg(x)
