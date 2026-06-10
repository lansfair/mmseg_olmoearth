from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv.transforms import BaseTransform
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample


def _collect_registries():
    registries = []
    try:
        from mmseg.registry import TRANSFORMS as MMSEG_TRANSFORMS
        registries.append(MMSEG_TRANSFORMS)
    except Exception:
        pass

    # Extra registration to the root MMEngine registry makes the transform more
    # tolerant when default_scope is missing or wrong.
    try:
        from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
        registries.append(MMENGINE_TRANSFORMS)
    except Exception:
        pass

    unique = []
    seen = set()
    for registry in registries:
        if id(registry) not in seen:
            unique.append(registry)
            seen.add(id(registry))
    return unique


def _register_transform(cls):
    for registry in _collect_registries():
        registry.register_module(module=cls, force=True)
    return cls


def _torch_load(path: Union[str, Path]):
    """torch.load wrapper compatible with different PyTorch versions."""
    try:
        return torch.load(str(path), map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location='cpu')


def _extract_tensor(obj: Any, preferred_keys: Sequence[str], path: Union[str, Path]) -> torch.Tensor:
    if torch.is_tensor(obj):
        return obj

    if isinstance(obj, dict):
        for key in preferred_keys:
            if key in obj and torch.is_tensor(obj[key]):
                return obj[key]

        tensor_values = [value for value in obj.values() if torch.is_tensor(value)]
        if len(tensor_values) == 1:
            return tensor_values[0]

    raise TypeError(
        f'Expected {path} to contain a Tensor, or a dict containing one Tensor. '
        f'Got type: {type(obj)}'
    )


@_register_transform
class LoadPastisSampleFromPT(BaseTransform):
    """Load one PASTIS S2 tensor and its segmentation target from .pt files.

    Input image shape:
        T x C x H x W, normally 12 x 13 x 64 x 64.

    Output image shape by default:
        C x H x W, because temporal_reduce='mean'.

    Important:
        DINOv3 ViT for Sentinel-2 normally expects a static 13-channel image.
        Therefore this loader reduces the 12 monthly images to one 13-channel
        image by default. You can change temporal_reduce if needed.
    """

    def __init__(
        self,
        temporal_reduce: str = 'mean',
        time_index: Optional[int] = None,
        to_float32: bool = True,
        img_scale_factor: Optional[float] = None,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        source_ignore_index: int = -1,
        target_ignore_index: Optional[int] = 255,
        cache_targets: bool = True,
    ) -> None:
        self.temporal_reduce = temporal_reduce
        self.time_index = time_index
        self.to_float32 = to_float32
        self.img_scale_factor = img_scale_factor
        self.mean = mean
        self.std = std
        self.source_ignore_index = source_ignore_index
        self.target_ignore_index = target_ignore_index
        self.cache_targets = cache_targets
        self._targets_cache: Dict[str, torch.Tensor] = {}

    def _reduce_temporal(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim == 3:
            return img

        if img.ndim != 4:
            raise ValueError(
                f'Image tensor must have shape T x C x H x W or C x H x W, '
                f'but got shape {tuple(img.shape)}.'
            )

        mode = self.temporal_reduce.lower()
        if mode == 'mean':
            return img.float().mean(dim=0)
        if mode == 'median':
            return img.float().median(dim=0).values
        if mode == 'max':
            return img.max(dim=0).values
        if mode == 'min':
            return img.min(dim=0).values
        if mode == 'first':
            return img[0]
        if mode == 'last':
            return img[-1]
        if mode == 'index':
            if self.time_index is None:
                raise ValueError("time_index must be set when temporal_reduce='index'.")
            return img[self.time_index]
        if mode == 'flatten':
            # Usually NOT recommended for DINOv3 SAT checkpoints, because it
            # changes input channels from 13 to T*13.
            t, c, h, w = img.shape
            return img.reshape(t * c, h, w)

        raise ValueError(
            f'Unsupported temporal_reduce={self.temporal_reduce!r}. Supported values: '
            "'mean', 'median', 'max', 'min', 'first', 'last', 'index', 'flatten'."
        )

    def _load_targets(self, targets_path: str) -> torch.Tensor:
        if self.cache_targets and targets_path in self._targets_cache:
            return self._targets_cache[targets_path]

        targets_obj = _torch_load(targets_path)
        targets = _extract_tensor(
            targets_obj,
            preferred_keys=('targets', 'target', 'mask', 'masks', 'labels', 'label'),
            path=targets_path,
        )

        if self.cache_targets:
            self._targets_cache[targets_path] = targets
        return targets

    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        if self.img_scale_factor is not None:
            img = img * float(self.img_scale_factor)

        if self.mean is not None:
            mean = torch.as_tensor(self.mean, dtype=img.dtype, device=img.device).view(-1, 1, 1)
            if mean.numel() != img.shape[0]:
                raise ValueError(f'mean has {mean.numel()} values, but image has {img.shape[0]} channels.')
            img = img - mean

        if self.std is not None:
            std = torch.as_tensor(self.std, dtype=img.dtype, device=img.device).view(-1, 1, 1)
            if std.numel() != img.shape[0]:
                raise ValueError(f'std has {std.numel()} values, but image has {img.shape[0]} channels.')
            img = img / std.clamp_min(1e-12)

        return img

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img_path = results['img_path']
        targets_path = results['targets_path']
        target_index = int(results['target_index'])

        img_obj = _torch_load(img_path)
        img = _extract_tensor(
            img_obj,
            preferred_keys=('img', 'image', 's2', 's2_image', 'data'),
            path=img_path,
        )

        img = self._reduce_temporal(img)
        if self.to_float32:
            img = img.float()
        img = self._normalize(img).contiguous()

        if img.ndim != 3:
            raise ValueError(
                f'After temporal reduction, image must be C x H x W, '
                f'but got shape {tuple(img.shape)}.'
            )

        targets = self._load_targets(targets_path)
        seg = targets[target_index]

        if seg.ndim == 3 and seg.shape[0] == 1:
            seg = seg[0]
        elif seg.ndim == 3 and seg.shape[-1] == 1:
            seg = seg[..., 0]

        if seg.ndim != 2:
            raise ValueError(
                f'One target should have shape H x W or 1 x H x W, '
                f'but got shape {tuple(seg.shape)} for target index {target_index}.'
            )

        seg = seg.long().contiguous()

        # PASTIS original ignore label is -1. MMSeg defaults commonly use 255
        # as ignore_index, so the default maps -1 -> 255. Set
        # target_ignore_index=-1 in config if you want to keep -1.
        if self.target_ignore_index is not None and self.target_ignore_index != self.source_ignore_index:
            seg = torch.where(
                seg == self.source_ignore_index,
                torch.as_tensor(self.target_ignore_index, dtype=seg.dtype, device=seg.device),
                seg,
            )

        h, w = img.shape[-2:]
        results['img'] = img
        results['gt_seg_map'] = seg
        results['ori_shape'] = (h, w)
        results['img_shape'] = (h, w)
        results['pad_shape'] = (h, w)
        results['seg_fields'] = ['gt_seg_map']
        results['num_channels'] = int(img.shape[0])
        return results


@_register_transform
class PastisResize(BaseTransform):
    """Resize image and segmentation mask.

    This is the reserved resize interface requested for PASTIS.

    Args:
        size: Target size as (height, width). If None, no fixed-size resize.
        scale_factor: Optional scale factor. Used only when size is None.
    """

    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        scale_factor: Optional[float] = None,
        interpolation: str = 'bilinear',
        align_corners: bool = False,
    ) -> None:
        if size is not None and scale_factor is not None:
            raise ValueError('Only one of size and scale_factor can be set.')

        if isinstance(size, int):
            size = (size, size)

        self.size = size
        self.scale_factor = scale_factor
        self.interpolation = interpolation
        self.align_corners = align_corners

    def _target_size(self, h: int, w: int) -> Optional[Tuple[int, int]]:
        if self.size is not None:
            return int(self.size[0]), int(self.size[1])

        if self.scale_factor is not None:
            return max(1, round(h * self.scale_factor)), max(1, round(w * self.scale_factor))

        return None

    def _interpolate_img(self, img: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        kwargs = dict(size=size, mode=self.interpolation)
        if self.interpolation in ('linear', 'bilinear', 'bicubic', 'trilinear'):
            kwargs['align_corners'] = self.align_corners
        return F.interpolate(img.unsqueeze(0), **kwargs).squeeze(0)

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img = results['img']
        seg = results['gt_seg_map']

        if img.ndim != 3:
            raise ValueError(f'PastisResize expects img as C x H x W, got {tuple(img.shape)}.')
        if seg.ndim != 2:
            raise ValueError(f'PastisResize expects gt_seg_map as H x W, got {tuple(seg.shape)}.')

        old_h, old_w = img.shape[-2:]
        target_size = self._target_size(old_h, old_w)
        if target_size is None:
            return results

        img = self._interpolate_img(img, target_size).contiguous()

        seg_dtype = seg.dtype
        seg = F.interpolate(
            seg.float().unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode='nearest',
        ).squeeze(0).squeeze(0).to(seg_dtype).contiguous()

        new_h, new_w = target_size
        results['img'] = img
        results['gt_seg_map'] = seg
        results['img_shape'] = (new_h, new_w)
        results['pad_shape'] = (new_h, new_w)
        results['scale_factor'] = (new_w / old_w, new_h / old_h)
        return results


@_register_transform
class PastisPackSegInputs(BaseTransform):
    """Pack PASTIS tensors into MMSegmentation model input format."""

    def __init__(
        self,
        meta_keys: Sequence[str] = (
            'img_path',
            'targets_path',
            'img_id',
            'sample_idx',
            'target_index',
            'split',
            'ori_shape',
            'img_shape',
            'pad_shape',
            'scale_factor',
            'num_channels',
        ),
    ) -> None:
        self.meta_keys = tuple(meta_keys)

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img = results['img']
        seg = results['gt_seg_map']

        if not torch.is_tensor(img):
            img = torch.as_tensor(img)
        if not torch.is_tensor(seg):
            seg = torch.as_tensor(seg)

        if img.ndim != 3:
            raise ValueError(f'inputs must be C x H x W, got {tuple(img.shape)}.')
        if seg.ndim != 2:
            raise ValueError(f'gt_seg_map must be H x W, got {tuple(seg.shape)}.')

        data_sample = SegDataSample()
        data_sample.gt_sem_seg = PixelData(data=seg.long().unsqueeze(0).contiguous())

        metainfo = {key: results[key] for key in self.meta_keys if key in results}
        data_sample.set_metainfo(metainfo)

        return dict(
            inputs=img.float().contiguous(),
            data_samples=data_sample,
        )
