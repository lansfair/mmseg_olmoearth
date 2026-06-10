from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from mmcv.transforms import BaseTransform
from mmengine.structures import PixelData
from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample


TEN_TO_THIRTEEN = (0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 8, 9)


def _torch_load(path: str):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _extract_tensor(value, keys: Sequence[str]) -> torch.Tensor:
    if isinstance(value, dict):
        for key in keys:
            if torch.is_tensor(value.get(key)):
                value = value[key]
                break
    if not torch.is_tensor(value):
        raise TypeError('Expected a tensor or a dict containing a tensor.')
    return value


@TRANSFORMS.register_module()
class LoadPastisSampleFromPT(BaseTransform):
    """Load the complete temporal sequence and its semantic target."""

    _target_cache: Dict[str, torch.Tensor] = {}

    def __init__(
        self,
        expected_times: Optional[int] = 12,
        expected_channels: Sequence[int] = (10, 13),
        source_ignore_index: int = -1,
        target_ignore_index: int = 255,
        to_float32: bool = True,
    ) -> None:
        self.expected_times = expected_times
        self.expected_channels = tuple(int(v) for v in expected_channels)
        self.source_ignore_index = int(source_ignore_index)
        self.target_ignore_index = int(target_ignore_index)
        self.to_float32 = bool(to_float32)

    @classmethod
    def _load_targets(cls, path: str) -> torch.Tensor:
        if path not in cls._target_cache:
            cls._target_cache[path] = _extract_tensor(
                _torch_load(path),
                ('targets', 'target', 'labels', 'label', 'masks', 'mask'),
            )
        return cls._target_cache[path]

    def transform(self, results: dict) -> dict:
        image = _extract_tensor(
            _torch_load(results['img_path']),
            ('image', 'img', 's2', 'data', 'tensor'),
        )
        if image.ndim != 4:
            raise ValueError(f'Expected image (T,C,H,W), got {tuple(image.shape)}.')
        if self.expected_times is not None and image.shape[0] != self.expected_times:
            raise ValueError(
                f'Expected T={self.expected_times}, got T={image.shape[0]} '
                f'in {results["img_path"]}.'
            )
        if image.shape[1] not in self.expected_channels:
            raise ValueError(
                f'Expected channels in {self.expected_channels}, got {image.shape[1]}.'
            )
        if self.to_float32:
            image = image.float()

        targets = self._load_targets(results['targets_path'])
        if targets.ndim != 3:
            raise ValueError(f'Expected targets (N,H,W), got {tuple(targets.shape)}.')
        target = targets[int(results['target_index'])].long().clone()
        target[target == self.source_ignore_index] = self.target_ignore_index

        height, width = map(int, image.shape[-2:])
        if tuple(target.shape[-2:]) != (height, width):
            raise ValueError(
                f'Image/target shape mismatch: {(height, width)} vs '
                f'{tuple(target.shape[-2:])}.'
            )

        results['img'] = image.contiguous()
        results['gt_seg_map'] = target.contiguous()
        results['source_shape'] = (height, width)
        results['ori_shape'] = (height, width)
        results['img_shape'] = (height, width)
        results['pad_shape'] = (height, width)
        results['num_times'] = int(image.shape[0])
        results['num_channels'] = int(image.shape[1])
        results['seg_fields'] = ['gt_seg_map']
        return results


@TRANSFORMS.register_module()
class PastisResize(BaseTransform):
    """Resize every time step with bilinear interpolation and mask by nearest."""

    def __init__(self, size: Optional[Sequence[int]] = None) -> None:
        if size is None:
            self.size = None
        elif isinstance(size, int):
            self.size = (int(size), int(size))
        elif len(size) == 2:
            self.size = (int(size[0]), int(size[1]))
        else:
            raise ValueError('size must be None, an int, or (height, width).')

    def transform(self, results: dict) -> dict:
        if self.size is None:
            return results
        image = results['img']
        target = results['gt_seg_map']
        image = F.interpolate(
            image, size=self.size, mode='bilinear', align_corners=False
        )
        target = F.interpolate(
            target[None, None].float(), size=self.size, mode='nearest'
        )[0, 0].long()
        results['img'] = image.contiguous()
        results['gt_seg_map'] = target.contiguous()
        # Evaluation uses the resized label, so ori_shape intentionally follows
        # the resized resolution. The pre-resize size remains in source_shape.
        results['ori_shape'] = self.size
        results['img_shape'] = self.size
        results['pad_shape'] = self.size
        return results


@TRANSFORMS.register_module()
class NormalizePastisFromJSON(BaseTransform):
    """Normalize Sentinel-2 data using train-fold statistics from JSON.

    The supplied file contains ten values per fold. They are expanded to the
    agreed 13-band order with mapping
    ``[0,0,1,2,3,4,5,6,7,7,8,8,9]``. If an input sample still has ten bands,
    the same mapping first expands the image itself to thirteen bands.
    """

    def __init__(
        self,
        stats_file: str,
        folds: Sequence[str] = ('Fold_1', 'Fold_2', 'Fold_3'),
        channel_map_10_to_13: Sequence[int] = TEN_TO_THIRTEEN,
        adapt_image_10_to_13: bool = True,
        eps: float = 1e-6,
    ) -> None:
        self.stats_file = str(Path(stats_file).expanduser())
        self.folds = tuple(str(fold) for fold in folds)
        self.channel_map = tuple(int(index) for index in channel_map_10_to_13)
        self.adapt_image_10_to_13 = bool(adapt_image_10_to_13)
        self.eps = float(eps)
        self.mean10, self.std10 = self._read_train_statistics()
        index = torch.tensor(self.channel_map, dtype=torch.long)
        self.mean13 = self.mean10.index_select(0, index)
        self.std13 = self.std10.index_select(0, index)

    def _read_train_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        path = Path(self.stats_file).resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Normalization JSON not found: {path}')
        data = json.loads(path.read_text(encoding='utf-8'))
        means, stds = [], []
        for fold in self.folds:
            if fold not in data:
                raise KeyError(f'{fold!r} not found in {path}.')
            mean = torch.tensor(data[fold]['mean'], dtype=torch.float32)
            std = torch.tensor(data[fold]['std'], dtype=torch.float32)
            if mean.numel() != 10 or std.numel() != 10:
                raise ValueError(f'{fold} must provide ten mean/std values.')
            means.append(mean)
            stds.append(std)
        # No per-fold pixel counts are available, so use the previously agreed
        # equal-fold arithmetic mean for both statistics.
        return torch.stack(means).mean(0), torch.stack(stds).mean(0)

    def transform(self, results: dict) -> dict:
        image = results['img'].float()
        channels = int(image.shape[1])
        if channels == 10 and self.adapt_image_10_to_13:
            index = torch.tensor(self.channel_map, device=image.device)
            image = image.index_select(1, index)
            channels = 13
        if channels == 13:
            mean, std = self.mean13, self.std13
        elif channels == 10:
            mean, std = self.mean10, self.std10
        else:
            raise ValueError(f'Normalization supports 10 or 13 channels, got {channels}.')
        mean = mean.to(image.device).view(1, channels, 1, 1)
        std = std.to(image.device).clamp_min(self.eps).view(1, channels, 1, 1)
        results['img'] = ((image - mean) / std).contiguous()
        results['num_channels'] = channels
        results['norm_stats_file'] = str(Path(self.stats_file).resolve())
        results['norm_folds'] = self.folds
        return results


@TRANSFORMS.register_module()
class PastisPackSegInputs(BaseTransform):
    """Pack a temporal tensor without converting it to HWC."""

    META_KEYS = (
        'img_path', 'targets_path', 'img_id', 'split', 'target_index',
        'source_shape', 'ori_shape', 'img_shape', 'pad_shape',
        'num_times', 'num_channels', 'norm_stats_file', 'norm_folds',
        'reduce_zero_label',
    )

    def __init__(self, meta_keys: Optional[Sequence[str]] = None) -> None:
        self.meta_keys = tuple(meta_keys) if meta_keys is not None else self.META_KEYS

    def transform(self, results: dict) -> dict:
        image = results['img']
        target = results['gt_seg_map']
        if not torch.is_tensor(image) or image.ndim != 4:
            raise ValueError('PastisPackSegInputs expects image tensor (T,C,H,W).')
        if not torch.is_tensor(target) or target.ndim != 2:
            raise ValueError('PastisPackSegInputs expects target tensor (H,W).')

        data_sample = SegDataSample()
        data_sample.gt_sem_seg = PixelData(data=target[None].long().contiguous())
        metainfo = {key: results[key] for key in self.meta_keys if key in results}
        data_sample.set_metainfo(metainfo)
        return dict(inputs=image.float().contiguous(), data_samples=data_sample)
