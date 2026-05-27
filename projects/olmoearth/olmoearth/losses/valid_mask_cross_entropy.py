from __future__ import annotations

import torch
from mmseg.registry import MODELS
from torch import Tensor, nn
from torch.nn import functional as F


@MODELS.register_module()
class ValidMaskCrossEntropyLoss(nn.Module):
    """Cross entropy over pixels selected by a valid mask.

    Custom OLMoEarth heads pass the per-sample valid mask to this loss after
    extracting it from ``SegDataSample``.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        ignore_index: int = 255,
        loss_name: str = "loss_valid_ce",
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name
        self.supports_valid_mask = True

    @property
    def loss_name(self) -> str:
        return self._loss_name

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        valid_mask: Tensor | None = None,
        ignore_index: int | None = None,
    ) -> Tensor:
        if ignore_index is None:
            ignore_index = self.ignore_index
        if valid_mask is None:
            return self.loss_weight * F.cross_entropy(
                pred, target, ignore_index=ignore_index
            )
        ce_target = target.masked_fill(target == ignore_index, 0)
        ce_target = ce_target.clamp(0, pred.shape[1] - 1)
        per_pixel = F.cross_entropy(pred, ce_target, reduction="none")
        valid_mask = valid_mask.float() * (target != ignore_index).float()
        denom = valid_mask.sum()
        numerator = (per_pixel * valid_mask).sum()
        loss = torch.where(
            denom > 0,
            numerator / denom.clamp(min=1.0),
            numerator,
        )
        return self.loss_weight * loss
