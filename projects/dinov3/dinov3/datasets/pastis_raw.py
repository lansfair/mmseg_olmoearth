from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from mmengine.dataset import Compose
from mmseg.registry import DATASETS

from projects.olmoearth.olmoearth.datasets.olmoearth_seg_dataset import (
    DATASET_METAINFO,
)


@DATASETS.register_module()
class DINOv3RawPASTISDataset:
    """Read original PASTIS-R with TorchGeo and output MMSeg samples.

    ``data_root`` may point either to the parent directory containing
    ``PASTIS-R`` or directly to the ``PASTIS-R`` directory.
    """

    METAINFO = DATASET_METAINFO["pastis"]

    def __init__(
        self,
        data_root: str | Path,
        folds: Sequence[int],
        pipeline: list[dict[str, Any]] | None = None,
        num_timesteps: int = 12,
        num_bands: int = 10,
        ignore_index: int = 255,
        source_ignore_values: Sequence[int] = (-1, 19),
        metainfo: dict[str, Any] | None = None,
        test_mode: bool = False,
        **kwargs,
    ) -> None:
        from torchgeo.datasets import PASTIS

        self.data_root = Path(data_root)
        self.torchgeo_root = self._torchgeo_root(self.data_root)
        self.folds = tuple(int(fold) for fold in folds)
        self.num_timesteps = int(num_timesteps)
        self.num_bands = int(num_bands)
        self.ignore_index = int(ignore_index)
        self.source_ignore_values = tuple(int(x) for x in source_ignore_values)
        self.metainfo = metainfo or self.METAINFO
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline or [])
        self.dataset = PASTIS(
            root=str(self.torchgeo_root),
            folds=self.folds,
            bands="s2",
            mode="semantic",
            download=False,
        )

    @staticmethod
    def _torchgeo_root(data_root: Path) -> Path:
        if data_root.name == "PASTIS-R":
            return data_root.parent
        return data_root

    def __len__(self) -> int:
        return len(self.dataset)

    def full_init(self) -> None:
        return

    @staticmethod
    def _sample_indices(num_available: int, num_timesteps: int) -> list[int]:
        if num_available <= 0:
            raise ValueError("PASTIS sample has no timesteps")
        n_dates = min(num_available, num_timesteps)
        indices = torch.linspace(0, num_available - 1, n_dates).long().tolist()
        while len(indices) < num_timesteps:
            indices.append(indices[-1])
        return indices

    @staticmethod
    def _tchw_to_hw_flat(image: np.ndarray) -> np.ndarray:
        if image.ndim != 4:
            raise ValueError(f"Expected image as TCHW, got {image.shape}")
        timesteps, channels, height, width = image.shape
        return image.transpose(2, 3, 1, 0).reshape(
            height,
            width,
            channels * timesteps,
        )

    def _prepare_results(self, idx: int) -> dict[str, Any]:
        sample = self.dataset[idx]
        image = sample["image"].float()
        if int(image.shape[1]) != self.num_bands:
            raise ValueError(
                f"Expected {self.num_bands} PASTIS S2 bands, "
                f"got {int(image.shape[1])} for sample {idx}"
            )
        indices = self._sample_indices(int(image.shape[0]), self.num_timesteps)
        image_np = image[indices].numpy().astype(np.float32, copy=False)

        label = sample["mask"].long().numpy().astype(np.int64).squeeze()
        label = label.copy()
        for value in self.source_ignore_values:
            label[label == value] = self.ignore_index

        return {
            "img": self._tchw_to_hw_flat(image_np),
            "gt_seg_map": label,
            "seg_fields": ["gt_seg_map"],
            "img_shape": tuple(label.shape),
            "ori_shape": tuple(label.shape),
            "sample_id": f"folds_{'-'.join(map(str, self.folds))}_{idx:06d}",
            "dataset_name": "pastis",
            "pastis_num_timesteps": self.num_timesteps,
            "pastis_num_bands": self.num_bands,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.pipeline(self._prepare_results(idx))
