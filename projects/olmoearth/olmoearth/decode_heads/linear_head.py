from __future__ import annotations

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from torch import Tensor, nn
from torch.nn import functional as F

from .valid_mask_mixin import ValidMaskLossMixin


@MODELS.register_module()
class OlmoEarthLinearHead(ValidMaskLossMixin, BaseDecodeHead):
    """Linear probe / fine-tuning head for dense OLMoEarth features."""

    def __init__(
        self,
        scale_factor: int = 4,
        use_valid_mask: bool = False,
        valid_mask_loss: bool = False,
        **kwargs,
    ) -> None:
        kwargs.setdefault("dropout_ratio", 0)
        super().__init__(input_transform=None, **kwargs)
        self.scale_factor = scale_factor
        self.use_valid_mask = use_valid_mask
        self.valid_mask_loss = valid_mask_loss
        self.classifier = nn.Conv2d(
            self.in_channels,
            self.num_classes,
            kernel_size=1,
        )

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
        return self.classifier(x)

    def loss_by_feat(
        self,
        seg_logits: Tensor,
        batch_data_samples: list,
    ) -> dict:
        seg_label = self._stack_batch_gt(batch_data_samples).squeeze(1).long()
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
