
import numpy as np
from PIL import Image
from typing import Dict

from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS

try:
    from osgeo import gdal
except ImportError:
    gdal = None


@TRANSFORMS.register_module()
class AddCopernicusMeta(BaseTransform):
    """Attach Copernicus-FM metadata to the sample metainfo."""

    def __init__(self, lon=np.nan, lat=np.nan, time=np.nan, patch_area=np.nan):
        self.meta = np.array([lon, lat, time, patch_area], dtype=np.float32)

    def transform(self, results):
        results['copernicus_meta'] = self.meta.copy()
        return results


@TRANSFORMS.register_module()
class NormalizeMultibandImage(BaseTransform):
    """Normalize a multi-band image before geometric augmentation."""

    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, -1)

    def transform(self, results):
        results['img'] = (results['img'].astype(np.float32) -
                          self.mean) / self.std
        return results

@TRANSFORMS.register_module()
class LoadSinglePNGImageFromFile(BaseTransform):
    """Load a Remote Sensing mage from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is a float64 array.
            Defaults to True.
    """

    def __init__(self, to_float32: bool = True):
        self.to_float32 = to_float32

        if gdal is None:
            raise RuntimeError('gdal is not installed')

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        img = np.array(Image.open(filename))

        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32})')
        return repr_str

@TRANSFORMS.register_module()
class LoadLocalPtsdamAnnotations(BaseTransform):
    """Load and remap original DFC2020 labels to 8 valid classes."""

    cls_mapping = {
        0: 255,
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 255
    }

    def __init__(self):
        if gdal is None:
            raise RuntimeError('gdal is not installed')

    def transform(self, results):
        ds = gdal.Open(results['seg_map_path'])
        if ds is None:
            raise FileNotFoundError(
                f'Unable to open file: {results["seg_map_path"]}')
        seg_map = ds.ReadAsArray()
        remapped = np.full(seg_map.shape, 255, dtype=np.uint8)
        for old_label, new_label in self.cls_mapping.items():
            remapped[seg_map == old_label] = new_label
        if results.get('label_map', None) is not None:
            remapped_copy = remapped.copy()
            for old_id, new_id in results['label_map'].items():
                remapped[remapped_copy == old_id] = new_id
        results['gt_seg_map'] = remapped
        results.setdefault('seg_fields', []).append('gt_seg_map')
        return results
