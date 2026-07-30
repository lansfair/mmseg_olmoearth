"""Data loading transforms for the DOFAv2 project."""

from __future__ import annotations

from typing import Any

import numpy as np
import rasterio
from mmcv.transforms import BaseTransform
from mmseg.datasets.transforms import LoadAnnotations
from mmseg.registry import TRANSFORMS

from .datasets import build_geobench_dataset


def _stat_value(stats: Any, key: str) -> float:
    if isinstance(stats, dict):
        return float(stats[key])
    return float(getattr(stats, key))


def _apply_label_rules(results: dict, label: np.ndarray) -> np.ndarray:
    label = np.asarray(label).astype(np.int64, copy=True)
    if results.get('reduce_zero_label', False):
        label = label.copy()
        label[label == 0] = 255
        label -= 1
        label[label == 254] = 255
    label_map = results.get('label_map')
    if label_map:
        original = label.copy()
        for old_id, new_id in label_map.items():
            label[original == old_id] = new_id
    return label


@TRANSFORMS.register_module()
class CenterCrop(BaseTransform):
    """Deterministically center-crop an image and its segmentation fields."""

    def __init__(self, crop_size: int | tuple[int, int]):
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        if len(crop_size) != 2 or min(crop_size) <= 0:
            raise ValueError('crop_size must contain two positive integers.')
        self.crop_size = tuple(crop_size)

    def transform(self, results: dict) -> dict:
        height, width = results['img'].shape[:2]
        crop_h, crop_w = self.crop_size
        if height < crop_h or width < crop_w:
            raise ValueError(
                f'Cannot center-crop {self.crop_size} from {(height, width)}.')
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
        slices = (slice(top, top + crop_h), slice(left, left + crop_w))
        results['img'] = np.ascontiguousarray(results['img'][slices])
        for key in results.get('seg_fields', []):
            results[key] = np.ascontiguousarray(results[key][slices])
        results['img_shape'] = self.crop_size
        return results


@TRANSFORMS.register_module()
class LoadDOFAGeoBenchSample(BaseTransform):
    """Load and normalize one GEO-Bench sample using task statistics."""

    def __init__(self,
                 num_classes: int = 7,
                 ignore_index: int = 255,
                 normalize: bool = True):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.normalize = normalize
        self._dataset = None
        self._dataset_key = None
        self._means = None
        self._stds = None

    def _get_dataset(self, results: dict):
        key = (
            results['dataset_root'],
            results['split'],
            results['partition_name'],
            tuple(results['band_names']),
            results['geobench_format'],
        )
        if self._dataset is not None and key == self._dataset_key:
            return self._dataset

        self._dataset = build_geobench_dataset(
            dataset_root=results['dataset_root'],
            split=results['split'],
            partition_name=results['partition_name'],
            band_names=results['band_names'],
            geobench_format=results['geobench_format'],
        )
        try:
            band_stats = self._dataset.band_stats
            self._means = np.asarray([
                _stat_value(band_stats[name], 'mean')
                for name in results['band_names']
            ], dtype=np.float32)
            self._stds = np.asarray([
                _stat_value(band_stats[name], 'std')
                for name in results['band_names']
            ], dtype=np.float32)
        except (AttributeError, KeyError, TypeError):
            # Older GEO-Bench releases expose the same statistics through the
            # dataset instead of task.band_stats.
            self._means, self._stds = self._dataset.normalization_stats()
            self._means = np.asarray(self._means, dtype=np.float32)
            self._stds = np.asarray(self._stds, dtype=np.float32)
        self._dataset_key = key
        return self._dataset

    def transform(self, results: dict) -> dict:
        dataset = self._get_dataset(results)
        sample = dataset[results['sample_idx']]
        image, _ = sample.pack_to_3d(
            band_names=results['band_names'],
            resample=True,
            resample_order=3,
        )
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 3:
            raise RuntimeError(
                f'Expected an HWC GEO-Bench image, got {image.shape}.')
        if image.shape[-1] != len(results['band_names']):
            raise RuntimeError(
                f'Expected {len(results["band_names"])} GEO-Bench bands, '
                f'got image shape {image.shape}.')
        if self.normalize:
            stds = np.maximum(self._stds, np.finfo(np.float32).eps)
            image = (
                image - self._means.reshape(1, 1, -1)
            ) / stds.reshape(1, 1, -1)

        label = np.asarray(sample.label.data)
        label = np.squeeze(label)
        if label.ndim != 2:
            raise RuntimeError(
                f'Expected a 2D GEO-Bench label, got {label.shape}.')
        if np.issubdtype(label.dtype, np.floating):
            invalid = ~np.isfinite(label)
            label = np.rint(label).astype(np.int64)
            label[invalid] = self.ignore_index
        label = _apply_label_rules(results, label)
        invalid = (label < 0) | (label >= self.num_classes)
        label[invalid] = self.ignore_index

        results['img'] = np.ascontiguousarray(image)
        results['gt_seg_map'] = np.ascontiguousarray(label)
        results['img_shape'] = image.shape[:2]
        results['ori_shape'] = image.shape[:2]
        results['seg_fields'] = ['gt_seg_map']
        results['sample_name'] = getattr(
            sample, 'sample_name', str(results['sample_idx']))
        return results


@TRANSFORMS.register_module()
class LoadImageFromTIF(BaseTransform):
    """Load a rasterio image in its file band order as HWC."""

    def __init__(self, to_float32: bool = True):
        self.to_float32 = to_float32

    def transform(self, results: dict) -> dict:
        with rasterio.open(results['img_path']) as source:
            image = np.moveaxis(source.read(), 0, -1)
        if self.to_float32:
            image = image.astype(np.float32)
        results['img'] = np.ascontiguousarray(image)
        results['img_shape'] = image.shape[:2]
        results['ori_shape'] = image.shape[:2]
        return results


@TRANSFORMS.register_module()
class LoadSegMapFromTIF(BaseTransform):
    """Load a single-band segmentation map with rasterio."""

    def transform(self, results: dict) -> dict:
        seg_map_path = results.get('seg_map_path')
        if not seg_map_path:
            raise KeyError('seg_map_path is required by LoadSegMapFromTIF.')
        with rasterio.open(seg_map_path) as source:
            label = source.read(1)
        label = _apply_label_rules(results, label)
        results['gt_seg_map'] = np.ascontiguousarray(label)
        results.setdefault('seg_fields', []).append('gt_seg_map')
        return results


@TRANSFORMS.register_module()
class LoadSVDTAnnotations(LoadAnnotations):
    """Load SVDT labels and map its foreground value 255 to class 1."""

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        label = np.asarray(results['gt_seg_map'])
        if label.ndim == 3:
            label = label[..., 0]
        label = label.astype(np.int64, copy=True)
        label[label == 255] = 1
        results['gt_seg_map'] = label
        return results
