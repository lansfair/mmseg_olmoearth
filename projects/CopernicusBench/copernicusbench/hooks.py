import os.path as osp
from typing import Optional, Sequence, Tuple

import mmcv
import numpy as np
from mmengine.fileio import get
from mmengine.dist import is_main_process, master_only
from mmengine.runner import Runner

from mmseg.engine.hooks import SegVisualizationHook
from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample

try:
    from osgeo import gdal
except ImportError:
    gdal = None

__all__ = ['CopernicusSegVisualizationHook']


@HOOKS.register_module()
class CopernicusSegVisualizationHook(SegVisualizationHook):
    """Visualization hook for multi-band Copernicus GeoTIFF inputs."""

    def __init__(self,
                 rgb_band_indices: Tuple[int, int, int] = (3, 2, 1),
                 percentile: Tuple[float, float] = (2., 98.),
                 draw_original: bool = True,
                 draw_gt: bool = True,
                 draw_pred: bool = True,
                 with_labels: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.rgb_band_indices = rgb_band_indices
        self.percentile = percentile
        self.draw_original = draw_original
        self.draw_gt = draw_gt
        self.draw_pred = draw_pred
        self.with_labels = with_labels

    def _stretch_to_uint8(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        out = np.zeros_like(image, dtype=np.float32)
        for channel in range(image.shape[-1]):
            band = image[..., channel]
            low, high = np.nanpercentile(band, self.percentile)
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                out[..., channel] = 0
                continue
            out[..., channel] = np.clip((band - low) / (high - low), 0, 1)
        return np.ascontiguousarray((out * 255).astype(np.uint8))

    def _read_geotiff_rgb(self, img_path: str) -> Optional[np.ndarray]:
        if gdal is None:
            return None
        ds = gdal.Open(img_path)
        if ds is None:
            return None

        bands = []
        for index in self.rgb_band_indices:
            band = ds.GetRasterBand(index + 1)
            if band is None:
                return None
            bands.append(band.ReadAsArray())
        if not bands:
            return None
        img = np.stack(bands, axis=-1)
        return self._stretch_to_uint8(np.nan_to_num(img))

    def _read_image(self, img_path: str) -> np.ndarray:
        img = self._read_geotiff_rgb(img_path)
        if img is not None:
            return img

        img_bytes = get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[-1] > 3:
            img = img[..., :3]
        return np.ascontiguousarray(img.astype(np.uint8, copy=False))

    def _draw_seg_overlay(self, image: np.ndarray,
                          sem_seg: SegDataSample) -> np.ndarray:
        palette = np.asarray(
            self._visualizer.dataset_meta.get('palette', []), dtype=np.uint8)
        if palette.size == 0:
            return image.copy()

        seg = sem_seg.cpu().data
        if hasattr(seg, 'numpy'):
            seg = seg.numpy()
        if seg.ndim == 3:
            seg = seg[0]
        seg = seg.astype(np.int64, copy=False)

        if seg.shape[:2] != image.shape[:2]:
            seg = mmcv.imresize(
                seg.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation='nearest').astype(np.int64)

        color_mask = np.zeros_like(image, dtype=np.uint8)
        valid = (seg >= 0) & (seg < len(palette))
        color_mask[valid] = palette[seg[valid]]

        out = image.astype(np.float32)
        out[valid] = out[valid] * (1 - self._visualizer.alpha) + \
            color_mask[valid].astype(np.float32) * self._visualizer.alpha
        return np.ascontiguousarray(out.astype(np.uint8))

    @master_only
    def _draw_datasample(self, name: str, image: np.ndarray,
                         data_sample: SegDataSample, step: int) -> None:
        classes = self._visualizer.dataset_meta.get('classes', None)
        palette = self._visualizer.dataset_meta.get('palette', None)
        drawn_images = []

        image = np.ascontiguousarray(image.astype(np.uint8, copy=False))
        if self.draw_original:
            drawn_images.append(image)
        if self.draw_gt and 'gt_sem_seg' in data_sample:
            gt_img = self._draw_seg_overlay(image.copy(),
                                            data_sample.gt_sem_seg)
            drawn_images.append(gt_img)
        if self.draw_pred and 'pred_sem_seg' in data_sample:
            pred_img = self._draw_seg_overlay(image.copy(),
                                              data_sample.pred_sem_seg)
            drawn_images.append(pred_img)

        drawn_img = np.concatenate(drawn_images, axis=1)
        if self.show:
            self._visualizer.show(
                drawn_img, win_name=name, wait_time=self.wait_time)
        else:
            self._visualizer.add_image(name, drawn_img, step)

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[SegDataSample]) -> None:
        if self.draw is False or not is_main_process():
            return

        total_curr_iter = runner.iter + batch_idx
        if total_curr_iter % self.interval != 0:
            return

        img_path = outputs[0].img_path
        img = self._read_image(img_path)
        window_name = f'val_{osp.basename(img_path)}'
        self._draw_datasample(window_name, img, outputs[0], total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[SegDataSample]) -> None:
        if self.draw is False or not is_main_process():
            return

        for data_sample in outputs:
            self._test_index += 1
            img_path = data_sample.img_path
            img = self._read_image(img_path)
            window_name = f'test_{osp.basename(img_path)}'
            self._draw_datasample(window_name, img, data_sample,
                                  self._test_index)
