from typing import Optional, Sequence, Tuple

import numpy as np
import torch
try:
    from mmcv.transforms import BaseTransform
except Exception:  # pragma: no cover
    class BaseTransform:
        def __call__(self, results):
            return self.transform(results)
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample


def _register_to_mmengine_transform(cls):
    try:
        from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
        if MMENGINE_TRANSFORMS.get(cls.__name__) is None:
            MMENGINE_TRANSFORMS.register_module(module=cls, force=True)
    except Exception:
        pass
    return cls


def _register_to_mmseg_transform(cls):
    try:
        from mmseg.registry import TRANSFORMS as MMSEG_TRANSFORMS
        if MMSEG_TRANSFORMS.get(cls.__name__) is None:
            MMSEG_TRANSFORMS.register_module(module=cls, force=True)
    except Exception:
        pass
    return cls


def register_transform(cls):
    cls = _register_to_mmengine_transform(cls)
    cls = _register_to_mmseg_transform(cls)
    return cls

S2_13_ORDER = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12')
PASTIS_10_ORDER = ('B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12')


def _extract_tensor(obj):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, dict):
        for key in ('img', 'image', 's2', 's2_image', 's2_images', 'data', 'x', 'inputs'):
            if key in obj:
                return _extract_tensor(obj[key])
    raise TypeError(f'Cannot extract tensor from object of type {type(obj)}')


def _safe_torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _impute_pastis10_to_s2_13(img: torch.Tensor) -> torch.Tensor:
    """Impute PASTIS 10 S2 bands to the Copernicus-FM 13-band order.

    Input order:  [B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12]
    Output order: [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12]

    This matches the PASTIS-R processor logic:
    B01 <- B02, B09 <- B8A, B10 <- B11.
    """
    return torch.stack([
        img[:, 0],  # B01 <- B02
        img[:, 0],  # B02
        img[:, 1],  # B03
        img[:, 2],  # B04
        img[:, 3],  # B05
        img[:, 4],  # B06
        img[:, 5],  # B07
        img[:, 6],  # B08
        img[:, 7],  # B8A
        img[:, 7],  # B09 <- B8A
        img[:, 8],  # B10 <- B11
        img[:, 8],  # B11
        img[:, 9],  # B12
    ], dim=1)


def _apply_channel_map(img: torch.Tensor, channel_map: Optional[str]) -> torch.Tensor:
    if channel_map is None or channel_map.lower() in ('none', 'identity'):
        return img
    channel_map = channel_map.lower()
    if img.ndim != 4:
        raise ValueError(f'Expected image shape T x C x H x W before channel mapping, got {tuple(img.shape)}')
    t, c, h, w = img.shape

    if channel_map in ('s2_13', 'already_s2_13'):
        if c != 13:
            raise ValueError(f'channel_map={channel_map} expects C=13, got C={c}')
        return img

    if channel_map in ('pastis10_to_s2_13', 'pastis10_impute_to_s2_13'):
        if c != 10:
            raise ValueError(f'{channel_map} expects C=10, got C={c}')
        return _impute_pastis10_to_s2_13(img)

    if channel_map == 'pastis10_padded_last_to_s2_13':
        if c != 13:
            raise ValueError(f'pastis10_padded_last_to_s2_13 expects C=13, got C={c}')
        # Assumption: [B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12,pad_B01,pad_B09,pad_B10]
        idx = [10, 0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 8, 9]
        return img[:, idx]

    raise ValueError(f'Unsupported channel_map={channel_map!r}')


@register_transform
class LoadPastisTemporalImageFromFile(BaseTransform):
    """Load one temporal Sentinel-2 tensor from ``s2_images/{idx}.pt``.

    The output ``results['img']`` is always T x C x H x W.
    """

    def __init__(
        self,
        to_float32: bool = True,
        scale_factor: Optional[float] = 10000.0,
        input_layout: str = 'TCHW',
        channel_map: Optional[str] = 's2_13',
    ):
        self.to_float32 = to_float32
        self.scale_factor = scale_factor
        self.input_layout = input_layout.upper()
        self.channel_map = channel_map

    def transform(self, results: dict) -> dict:
        obj = _safe_torch_load(results['img_path'])
        img = _extract_tensor(obj)
        if img.ndim == 3:
            img = img.unsqueeze(0)
        if img.ndim != 4:
            raise ValueError(f'Expected image tensor with 4 dims, got {tuple(img.shape)} from {results["img_path"]}')

        if self.input_layout == 'CTHW':
            img = img.permute(1, 0, 2, 3).contiguous()
        elif self.input_layout != 'TCHW':
            raise ValueError(f'Unsupported input_layout={self.input_layout!r}. Use TCHW or CTHW.')

        if self.to_float32:
            img = img.float()
        if self.scale_factor is not None:
            img = img / float(self.scale_factor)
        img = _apply_channel_map(img, self.channel_map)

        _, c, h, w = img.shape
        results['img'] = img.contiguous()
        results['ori_shape'] = (h, w)
        results['img_shape'] = (h, w)
        results['pad_shape'] = (h, w)
        results['num_channels'] = c
        results['num_timesteps'] = img.shape[0]
        return results


@register_transform
class LoadPastisTemporalAnnotations(BaseTransform):
    def transform(self, results: dict) -> dict:
        if 'gt_seg_map' not in results:
            targets = _safe_torch_load(results['target_path'])
            if isinstance(targets, dict):
                for key in ('targets', 'target', 'labels', 'masks', 'y'):
                    if key in targets:
                        targets = targets[key]
                        break
            results['gt_seg_map'] = targets[results['target_index']]
        results['gt_seg_map'] = torch.as_tensor(results['gt_seg_map']).long()
        results.setdefault('seg_fields', []).append('gt_seg_map')
        return results


@register_transform
class LoadPastisMonthsFromFile(BaseTransform):
    """Optional loader for ``months.pt``.

    It is not required by default because the first version sends NaN metadata to
    Copernicus-FM. The transform is included for later experiments with real time
    metadata.
    """

    def __init__(self, key: str = 'months'):
        self.key = key

    def transform(self, results: dict) -> dict:
        months_path = results.get('months_path', None)
        if months_path is None:
            return results
        months = _safe_torch_load(months_path)
        if isinstance(months, dict):
            months = months[self.key]
        results[self.key] = torch.as_tensor(months[results['target_index']]).long()
        return results


@register_transform
class PackPastisSegInputs(BaseTransform):
    def __init__(
        self,
        meta_keys: Tuple[str, ...] = (
            'img_path', 'target_path', 'months_path', 'target_index', 'sample_idx',
            'ori_shape', 'img_shape', 'pad_shape', 'num_channels', 'num_timesteps'
        ),
    ):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        img = results['img']
        if not torch.is_tensor(img):
            img = torch.as_tensor(img)
        img = img.contiguous()

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            gt = torch.as_tensor(results['gt_seg_map']).long()
            if gt.ndim == 2:
                gt = gt.unsqueeze(0)
            data_sample.gt_sem_seg = PixelData(data=gt.contiguous())

        img_meta = {key: results[key] for key in self.meta_keys if key in results}
        data_sample.set_metainfo(img_meta)
        return dict(inputs=img, data_samples=data_sample)
