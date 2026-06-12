from pathlib import Path
import os
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class LocalSVDTDataset(BaseSegDataset):
    """DFC2020 S2 semantic segmentation dataset for MMSegmentation.

    The original Copernicus-Bench split CSVs store label filenames under
    ``dfc/``. Sentinel-2 images use the same filename with ``dfc`` replaced by
    ``s2``.
    """

    METAINFO = dict(
        classes=(
            'Background',
            'Cropland',
        ),
        palette=[
            [0, 100, 0],
            [255, 187, 34],
        ],
    )

    def __init__(
        self,
        ann_file: str = '',
        data_prefix: dict = dict(img_path='img_dir', seg_map_path='ann_dir'),
        img_suffix='.tif',
        seg_map_suffix='.png',
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
        data_list = []
        txt_file = os.path.join(self.data_root, self.ann_file)
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_path = os.path.join(self.data_root, 'img_dir', line + '.png')
                ann_path = os.path.join(self.data_root, 'ann_dir', line + '_mask_seg.png')
                data_list.append(
                    dict(
                        img_path=img_path,
                        seg_map_path=ann_path,
                        label_map=self.label_map,
                        reduce_zero_label=self.reduce_zero_label,
                        seg_fields=[],
                    ))
        return data_list
