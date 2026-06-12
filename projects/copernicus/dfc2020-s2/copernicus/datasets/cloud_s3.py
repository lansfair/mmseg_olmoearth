from pathlib import Path

from mmseg.registry import DATASETS

from .base import CoBenchSegDataset


@DATASETS.register_module()
class CloudS3Dataset(CoBenchSegDataset):
    """Copernicus-Bench Cloud S3 semantic segmentation dataset."""

    METAINFO = dict(
        classes=('clear', 'cloud_sure', 'cloud_ambiguous', 'cloud_shadow',
                 'snow_ice'),
        palette=[
            [0, 0, 0],
            [255, 255, 255],
            [160, 160, 160],
            [80, 80, 80],
            [120, 180, 255],
        ])

    def __init__(
        self,
        ann_file='',
        data_prefix=dict(img_path='s3_olci', seg_map_path='cloud_multi'),
        ignore_index=255,
        reduce_zero_label=False,
        **kwargs,
    ):
        super().__init__(
            ann_file=ann_file,
            data_prefix=data_prefix,
            img_suffix='',
            seg_map_suffix='',
            ignore_index=ignore_index,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

    def load_data_list(self):
        img_dir = Path(self.data_prefix['img_path'])
        ann_dir = Path(self.data_prefix['seg_map_path'])
        data_list = []
        with open(self.ann_file, 'r', encoding='utf-8') as f:
            for line in f:
                filename = line.strip()
                if not filename:
                    continue
                data_list.append(
                    self._data_info(img_dir / filename, ann_dir / filename))
        return data_list
