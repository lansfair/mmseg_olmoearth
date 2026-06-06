from pathlib import Path

from mmseg.registry import DATASETS

from .base import CoBenchSegDataset


@DATASETS.register_module()
class CloudS2Dataset(CoBenchSegDataset):
    """Copernicus-Bench Cloud S2 semantic segmentation dataset."""

    METAINFO = dict(
        classes=('clear', 'thick_cloud', 'thin_cloud', 'cloud_shadow'),
        palette=[
            [0, 0, 0],
            [255, 255, 255],
            [160, 160, 160],
            [80, 80, 80],
        ])

    def __init__(
        self,
        ann_file='',
        data_prefix=dict(img_path='s2_toa', seg_map_path='cloud'),
        img_suffix='.tif',
        seg_map_suffix='.tif',
        ignore_index=255,
        reduce_zero_label=False,
        **kwargs,
    ):
        super().__init__(
            ann_file=ann_file,
            data_prefix=data_prefix,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

    def load_data_list(self):
        img_dir = Path(self.data_prefix['img_path'])
        ann_dir = Path(self.data_prefix['seg_map_path'])
        data_list = []
        with open(self.ann_file, 'r', encoding='utf-8') as f:
            for line in f:
                stem = line.strip()
                if not stem:
                    continue
                data_list.append(
                    self._data_info(img_dir / f'{stem}{self.img_suffix}',
                                    ann_dir / f'{stem}{self.seg_map_suffix}'))
        return data_list
