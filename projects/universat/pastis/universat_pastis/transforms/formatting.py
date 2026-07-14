"""Packing transform for UniverSat PASTIS-R inputs."""

from typing import Sequence, Tuple

import torch
from mmcv.transforms import BaseTransform
from mmengine.structures import PixelData
from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample


@TRANSFORMS.register_module()
class PackUniverSatPASTISInputs(BaseTransform):
    """Pack a PASTIS-R sample into MMSegmentation inputs.

    Required keys:
        - ``{modality}`` tensors of shape ``(T, C, H, W)``
        - ``{modality}_dates`` tensors of shape ``(T,)``
        - ``gt_seg_map``

    Added keys:
        - ``inputs`` (dict): modality tensors and date tensors
        - ``data_samples`` (SegDataSample)
    """

    def __init__(
        self,
        modalities: Sequence[str] = ("s2", "s1"),
        meta_keys: Tuple[str, ...] = (
            "id_patch",
            "img_shape",
            "ori_shape",
            "pad_shape",
        ),
    ):
        self.modalities = list(modalities)
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        inputs = {}
        for mod in self.modalities:
            inputs[mod] = results[mod]
            inputs[f"{mod}_dates"] = results[f"{mod}_dates"]

        data_sample = SegDataSample()
        gt = torch.as_tensor(results["gt_seg_map"]).long()
        if gt.ndim == 2:
            gt = gt.unsqueeze(0)
        data_sample.gt_sem_seg = PixelData(data=gt)

        img_meta = {key: results.get(key) for key in self.meta_keys if key in results}
        data_sample.set_metainfo(img_meta)

        return dict(inputs=inputs, data_samples=data_sample)
