from pathlib import Path

from mmseg.registry import DATASETS

from .base import CoBenchSegDataset


@DATASETS.register_module()
class LC100SegS3Dataset(CoBenchSegDataset):
    """Copernicus-Bench LC100 S3 static semantic segmentation dataset."""

    METAINFO = dict(
        classes=(
            'unknown',
            'shrubs',
            'herbaceous_vegetation',
            'cultivated_and_managed_vegetation',
            'urban_built_up',
            'bare_sparse_vegetation',
            'snow_ice',
            'permanent_water',
            'herbaceous_wetland',
            'moss_lichen',
            'closed_forest_evergreen_needle_leaf',
            'closed_forest_evergreen_broad_leaf',
            'closed_forest_deciduous_needle_leaf',
            'closed_forest_deciduous_broad_leaf',
            'closed_forest_mixed',
            'closed_forest_unknown',
            'open_forest_evergreen_needle_leaf',
            'open_forest_evergreen_broad_leaf',
            'open_forest_deciduous_needle_leaf',
            'open_forest_deciduous_broad_leaf',
            'open_forest_mixed',
            'open_forest_unknown',
            'open_sea',
        ),
        palette=[
            [0, 0, 0],
            [255, 187, 34],
            [255, 255, 76],
            [240, 150, 255],
            [250, 0, 0],
            [180, 180, 180],
            [240, 240, 240],
            [0, 100, 200],
            [0, 150, 160],
            [230, 220, 170],
            [0, 100, 0],
            [0, 160, 0],
            [170, 200, 0],
            [0, 120, 80],
            [40, 120, 40],
            [20, 80, 20],
            [120, 160, 0],
            [0, 180, 80],
            [200, 220, 0],
            [70, 160, 110],
            [90, 140, 60],
            [60, 100, 40],
            [0, 80, 180],
        ])

    def __init__(
        self,
        ann_file='',
        data_prefix=dict(img_path='s3_olci', seg_map_path='lc100'),
        ignore_index=255,
        reduce_zero_label=False,
        **kwargs,
    ):
        super().__init__(
            ann_file=ann_file,
            data_prefix=data_prefix,
            img_suffix='',
            seg_map_suffix='.tif',
            ignore_index=ignore_index,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

    def load_data_list(self):
        img_dir = Path(self.data_prefix['img_path'])
        ann_dir = Path(self.data_prefix['seg_map_path'])
        static_img = {}
        with open(self.ann_file, 'r', encoding='utf-8') as f:
            for line in f:
                row = line.strip()
                if not row:
                    continue
                patch_id, img_name = [
                    item.strip() for item in row.split(',', maxsplit=1)
                ]
                static_img[patch_id] = img_name
        data_list = []
        for patch_id, img_name in static_img.items():
            data_list.append(
                self._data_info(img_dir / patch_id / img_name,
                                ann_dir / f'{patch_id}{self.seg_map_suffix}'))
        return data_list
