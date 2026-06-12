import os.path as osp
from typing import Optional, Sequence, Tuple

import mmcv
import numpy as np
from mmengine.fileio import get
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
                 **kwargs):
        super().__init__(**kwargs)
        self.rgb_band_indices = rgb_band_indices
        self.percentile = percentile

    def _stretch_to_uint8(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        image = np.nan_to_num(image)
        out = np.zeros_like(image, dtype=np.float32)
        
        for channel in range(image.shape[-1]):
            band = image[..., channel]
            low, high = np.nanpercentile(band, self.percentile)
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                out[..., channel] = 0
                continue
            out[..., channel] = np.clip((band - low) / (high - low), 0, 1)
        
        img = (out * 255).astype(np.uint8)
        return np.ascontiguousarray(img)  # 强制连续

    def _read_geotiff_rgb(self, img_path: str) -> Optional[np.ndarray]:
        if gdal is None:
            return None
        ds = gdal.Open(img_path)
        if ds is None:
            return None

        img = ds.ReadAsArray()
        del ds  # 释放资源

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))  # (H, W, C)
            if img.shape[-1] >= 3:
                img = img[..., list(self.rgb_band_indices)]
            else:
                img = np.repeat(img[..., :1], 3, axis=-1)
        else:
            return None

        return self._stretch_to_uint8(img)

    def _read_image(self, img_path: str) -> np.ndarray:
        img = self._read_geotiff_rgb(img_path)
        if img is not None:
            return img

        # 普通图像读取
        img_bytes = get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[-1] > 3:
            img = img[..., :3]
        
        return np.ascontiguousarray(img)  # 强制连续

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[SegDataSample]) -> None:
        if not self.draw:
            return

        total_curr_iter = runner.iter + batch_idx
        if total_curr_iter % self.interval != 0:
            return

        img_path = outputs[0].img_path
        img = self._read_image(img_path)
        window_name = f'val_{osp.basename(img_path)}'
        
        self._visualizer.add_datasample(
            window_name,
            img,
            data_sample=outputs[0],
            draw_gt=False,
            draw_pred=True,
            show=self.show,
            wait_time=self.wait_time,
            step=total_curr_iter,
            with_labels=False
        )

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[SegDataSample]) -> None:
        if not self.draw:
            return

        for data_sample in outputs:
            self._test_index += 1
            img_path = data_sample.img_path
            img = self._read_image(img_path)
            window_name = f'test_{osp.basename(img_path)}'
            
            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=data_sample,
                draw_gt=True,
                draw_pred=True,
                show=self.show,
                wait_time=self.wait_time,
                step=self._test_index,
                with_labels=False
            )