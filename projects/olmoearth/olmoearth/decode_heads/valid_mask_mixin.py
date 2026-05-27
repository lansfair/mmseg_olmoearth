from __future__ import annotations

import torch
from mmseg.models.losses import accuracy
from torch import Tensor, nn


class ValidMaskLossMixin:
    """Shared valid-mask loss and logging helpers for OLMoEarth heads."""

    use_valid_mask: bool
    valid_mask_loss: bool
    ignore_index: int
    loss_decode: nn.Module | nn.ModuleList

    def _valid_mask(
        self,
        batch_data_samples: list,
        seg_label: Tensor,
    ) -> Tensor | None:
        if not self.use_valid_mask:
            return None
        masks = []
        for sample in batch_data_samples:
            if hasattr(sample, "gt_valid_mask"):
                masks.append(sample.gt_valid_mask.data.squeeze(0).float())
            else:
                masks.append(
                    (
                        sample.gt_sem_seg.data.squeeze(0)
                        != self.ignore_index
                    ).float()
                )
        return torch.stack(masks, dim=0).to(seg_label.device)

    def _configured_loss(
        self,
        seg_logits: Tensor,
        seg_label: Tensor,
        valid_mask: Tensor | None = None,
    ) -> dict:
        losses_decode = (
            self.loss_decode
            if isinstance(self.loss_decode, nn.ModuleList)
            else [self.loss_decode]
        )
        losses = {}
        for loss_decode in losses_decode:
            loss_name = loss_decode.loss_name
            if valid_mask is not None:
                if not getattr(loss_decode, "supports_valid_mask", False):
                    raise TypeError(
                        "valid_mask_loss=True requires a loss module that "
                        "sets supports_valid_mask=True, such as "
                        "ValidMaskCrossEntropyLoss."
                    )
                loss_value = loss_decode(
                    seg_logits,
                    seg_label,
                    valid_mask=valid_mask,
                    ignore_index=self.ignore_index,
                )
            else:
                loss_value = loss_decode(
                    seg_logits,
                    seg_label,
                    ignore_index=self.ignore_index,
                )
            if loss_name in losses:
                losses[loss_name] += loss_value
            else:
                losses[loss_name] = loss_value
        return losses

    def _accuracy(
        self,
        seg_logits: Tensor,
        seg_label: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        if valid_mask is None:
            return accuracy(
                seg_logits,
                seg_label,
                ignore_index=self.ignore_index,
            )
        valid = valid_mask.bool() & (seg_label != self.ignore_index)
        total = valid.sum()
        pred = seg_logits.argmax(dim=1)
        correct = ((pred == seg_label) & valid).sum().float()
        return torch.where(
            total > 0,
            correct / total.float().clamp(min=1.0),
            seg_logits.sum() * 0,
        )

    def _losses_with_optional_valid_mask(
        self,
        seg_logits: Tensor,
        seg_label: Tensor,
        batch_data_samples: list,
    ) -> dict:
        valid_mask = self._valid_mask(batch_data_samples, seg_label)
        if valid_mask is None or not self.valid_mask_loss:
            losses = self._configured_loss(seg_logits, seg_label)
        else:
            losses = self._configured_loss(seg_logits, seg_label, valid_mask)
        losses["acc_seg"] = self._accuracy(seg_logits, seg_label, valid_mask)
        return losses
