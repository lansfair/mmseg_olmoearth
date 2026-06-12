import os
from datetime import date, datetime

import numpy as np
from mmcv.transforms import BaseTransform

from mmseg.registry import TRANSFORMS

try:
    from osgeo import gdal, osr
except ImportError:
    gdal = None
    osr = None


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
class LoadCopernicusGeoTiffImageFromFile(BaseTransform):
    """Load a multi-band GeoTIFF and build Copernicus-FM metadata."""

    def __init__(self,
                 band_indices=None,
                 band_scales=None,
                 nan_to_num=True,
                 to_float32=True,
                 date_separator=None,
                 date_token_index=1,
                 patch_area=np.nan):
        if gdal is None:
            raise RuntimeError('gdal is not installed')
        self.band_indices = band_indices
        self.band_scales = None if band_scales is None else np.array(
            band_scales, dtype=np.float32)
        self.nan_to_num = nan_to_num
        self.to_float32 = to_float32
        self.date_separator = date_separator
        self.date_token_index = date_token_index
        self.patch_area = patch_area

    def _read_image(self, ds):
        if self.band_indices is None:
            img = ds.ReadAsArray()
            if img.ndim == 2:
                img = img[None, ...]
        else:
            bands = []
            for index in self.band_indices:
                band = ds.GetRasterBand(index)
                if band is None:
                    raise ValueError(
                        f'Band index {index} is out of range for '
                        f'{ds.GetDescription()}')
                bands.append(band.ReadAsArray())
            img = np.stack(bands)

        if self.to_float32:
            img = img.astype(np.float32)
        if self.nan_to_num:
            img = np.nan_to_num(img)
        if self.band_scales is not None:
            if len(self.band_scales) != img.shape[0]:
                raise ValueError(
                    'band_scales length must match the number of loaded '
                    f'bands, but got {len(self.band_scales)} scales for '
                    f'{img.shape[0]} bands.')
            img = img * self.band_scales.reshape(-1, 1, 1)
        return np.transpose(img, (1, 2, 0))

    def _read_lon_lat(self, ds):
        geotransform = ds.GetGeoTransform(can_return_null=True)
        if geotransform is None:
            return np.nan, np.nan

        center_x = (ds.RasterXSize - 1) / 2
        center_y = (ds.RasterYSize - 1) / 2
        x = (geotransform[0] + center_x * geotransform[1] +
             center_y * geotransform[2])
        y = (geotransform[3] + center_x * geotransform[4] +
             center_y * geotransform[5])

        projection = ds.GetProjection()
        if projection and osr is not None:
            src = osr.SpatialReference()
            if src.ImportFromWkt(projection) == 0:
                dst = osr.SpatialReference()
                dst.ImportFromEPSG(4326)
                if hasattr(src, 'SetAxisMappingStrategy'):
                    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                if not src.IsSame(dst):
                    x, y, _ = osr.CoordinateTransformation(
                        src, dst).TransformPoint(x, y)
        return x, y

    def _read_time(self, filename):
        if self.date_separator is None:
            return np.nan
        basename = os.path.basename(filename)
        try:
            token = basename.split(self.date_separator)[self.date_token_index]
            sensing_date = datetime.strptime(token[:8], '%Y%m%d').date()
        except (IndexError, ValueError):
            return np.nan
        return (sensing_date - date(1970, 1, 1)).days

    def transform(self, results):
        filename = results['img_path']
        ds = gdal.Open(filename)
        if ds is None:
            raise FileNotFoundError(f'Unable to open file: {filename}')

        img = self._read_image(ds)
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]

        lon, lat = self._read_lon_lat(ds)
        sensing_time = self._read_time(filename)
        results['copernicus_meta'] = np.array(
            [lon, lat, sensing_time, self.patch_area], dtype=np.float32)
        return results


@TRANSFORMS.register_module()
class LoadCoBenchSegAnnotations(BaseTransform):
    """Load segmentation labels with optional class-id remapping."""

    def __init__(self, label_mapping=None, default_value=None):
        if gdal is None:
            raise RuntimeError('gdal is not installed')
        self.label_mapping = label_mapping
        self.default_value = default_value

    def transform(self, results):
        ds = gdal.Open(results['seg_map_path'])
        if ds is None:
            raise FileNotFoundError(
                f'Unable to open file: {results["seg_map_path"]}')
        seg_map = ds.ReadAsArray()
        if seg_map.ndim == 3:
            seg_map = seg_map[0]

        if self.label_mapping is not None:
            if self.default_value is None:
                remapped = seg_map.copy()
            else:
                remapped = np.full(
                    seg_map.shape, self.default_value, dtype=np.int64)
            for old_label, new_label in self.label_mapping.items():
                remapped[seg_map == int(old_label)] = int(new_label)
            seg_map = remapped

        if results.get('label_map', None) is not None:
            remapped = seg_map.copy()
            for old_id, new_id in results['label_map'].items():
                seg_map[remapped == old_id] = new_id

        results['gt_seg_map'] = seg_map.astype(np.uint8)
        results.setdefault('seg_fields', []).append('gt_seg_map')
        return results


@TRANSFORMS.register_module()
class LoadDFC2020Annotations(BaseTransform):
    """Load and remap original DFC2020 labels to 8 valid classes."""

    cls_mapping = {
        0: 255,
        1: 0,
        2: 1,
        3: 255,
        4: 2,
        5: 3,
        6: 4,
        7: 5,
        8: 255,
        9: 6,
        10: 7,
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
