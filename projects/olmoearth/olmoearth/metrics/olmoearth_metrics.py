from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Sequence

import torch
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS


def _sample_get(sample, key: str):
    if isinstance(sample, dict):
        return sample.get(key)
    return getattr(sample, key, None)


def _pixel_data_tensor(value) -> torch.Tensor:
    if isinstance(value, dict):
        value = value["data"]
    elif hasattr(value, "data"):
        value = value.data
    return value.squeeze().long()


def _valid_mask_tensor(
    sample,
    gt: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    valid = _sample_get(sample, "gt_valid_mask")
    if valid is None:
        return gt != ignore_index
    if isinstance(valid, dict):
        valid = valid["data"]
    elif hasattr(valid, "data"):
        valid = valid.data
    return valid.squeeze().to(dtype=torch.bool)


@METRICS.register_module()
class OlmoEarthIoUMetric(BaseMetric):
    """OLMoEarth-style segmentation metric.

    This follows the pretraining eval path: filter ignored pixels, build a
    confusion matrix, and average IoU only over classes whose union is nonzero.
    """

    default_prefix = None

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        use_valid_mask: bool = False,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.use_valid_mask = use_valid_mask

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            pred = _pixel_data_tensor(_sample_get(sample, "pred_sem_seg"))
            gt = _pixel_data_tensor(_sample_get(sample, "gt_sem_seg"))
            valid = gt != self.ignore_index
            if self.use_valid_mask:
                valid = valid & _valid_mask_tensor(
                    sample,
                    gt,
                    self.ignore_index,
                )
            self.results.append(
                {"pred": pred.cpu(), "gt": gt.cpu(), "valid": valid.cpu()}
            )

    def compute_metrics(self, results: list[dict]) -> OrderedDict:
        confusion = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.float64
        )
        for result in results:
            pred = result["pred"].long()
            gt = result["gt"].long()
            valid = result["valid"].bool()
            pred = pred[valid]
            gt = gt[valid]
            in_range = (gt >= 0) & (gt < self.num_classes)
            pred = pred[in_range].clamp(0, self.num_classes - 1)
            gt = gt[in_range]
            if gt.numel() == 0:
                continue
            bincount = torch.bincount(
                self.num_classes * gt + pred,
                minlength=self.num_classes**2,
            ).reshape(self.num_classes, self.num_classes)
            confusion += bincount.to(confusion.dtype)

        tp = confusion.diag()
        fp = confusion.sum(dim=0) - tp
        fn = confusion.sum(dim=1) - tp
        union = tp + fp + fn
        iou = tp / (union + 1e-8)
        valid_classes = union > 0
        miou = iou[valid_classes].mean().item() if valid_classes.any() else 0.0

        total = confusion.sum()
        overall_acc = (tp.sum() / (total + 1e-8)).item()
        class_totals = tp + fn
        per_class_acc = tp / (class_totals + 1e-8)
        valid_acc_classes = class_totals > 0
        macro_acc = (
            per_class_acc[valid_acc_classes].mean().item()
            if valid_acc_classes.any()
            else 0.0
        )
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        per_class_f1 = 2 * precision * recall / (precision + recall + 1e-8)
        valid_f1_classes = class_totals > 0
        macro_f1 = (
            per_class_f1[valid_f1_classes].mean().item()
            if valid_f1_classes.any()
            else 0.0
        )
        micro_f1 = (
            2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum() + 1e-8)
        ).item()
        return OrderedDict(
            mIoU=miou,
            overall_acc=overall_acc,
            macro_acc=macro_acc,
            macro_f1=macro_f1,
            micro_f1=micro_f1,
        )


@METRICS.register_module()
class OlmoEarthAccuracyMetric(BaseMetric):
    """Micro accuracy over ignore-filtered and optionally valid-mask pixels."""

    default_prefix = None

    def __init__(
        self,
        ignore_index: int = 255,
        use_valid_mask: bool = True,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index
        self.use_valid_mask = use_valid_mask

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            pred = _pixel_data_tensor(_sample_get(sample, "pred_sem_seg"))
            gt = _pixel_data_tensor(_sample_get(sample, "gt_sem_seg"))
            valid = gt != self.ignore_index
            if self.use_valid_mask:
                valid = valid & _valid_mask_tensor(
                    sample,
                    gt,
                    self.ignore_index,
                )
            correct = ((pred == gt) & valid).sum().item()
            total = valid.sum().item()
            self.results.append({"correct": correct, "total": total})

    def compute_metrics(self, results: list[dict]) -> OrderedDict:
        total = sum(result["total"] for result in results)
        correct = sum(result["correct"] for result in results)
        accuracy = correct / total if total else 0.0
        return OrderedDict(accuracy=accuracy)
