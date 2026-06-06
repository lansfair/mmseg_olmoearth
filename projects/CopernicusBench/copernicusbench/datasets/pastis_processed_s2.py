from pathlib import Path

import torch
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class PASTISProcessedS2Dataset(BaseSegDataset):
    """OlmoEarth-processed PASTIS-R Sentinel-2 semantic dataset.

    Expected layout:

    - pastis_r_train/s2_images/{idx}.pt
    - pastis_r_train/targets.pt
    - pastis_r_train/months.pt

    and the same for ``valid`` and ``test``.
    """

    METAINFO = dict(
        classes=(
            'Background',
            'Meadow',
            'Soft winter wheat',
            'Corn',
            'Winter barley',
            'Winter rapeseed',
            'Spring barley',
            'Sunflower',
            'Grapevine',
            'Beet',
            'Winter triticale',
            'Winter durum wheat',
            'Fruits, vegetables, flowers',
            'Potatoes',
            'Leguminous fodder',
            'Soybeans',
            'Orchard',
            'Mixed cereal',
            'Sorghum',
        ),
        palette=[
            [0, 0, 0],
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 190],
            [0, 128, 128],
            [230, 190, 255],
            [170, 110, 40],
            [255, 250, 200],
            [128, 0, 0],
            [170, 255, 195],
            [128, 128, 0],
            [255, 215, 180],
        ],
    )

    def __init__(
        self,
        split='train',
        data_prefix=dict(img_path='', seg_map_path=''),
        ignore_index=255,
        reduce_zero_label=False,
        **kwargs,
    ):
        if split not in ('train', 'valid', 'test'):
            raise ValueError('split must be one of: train, valid, test.')
        self.split = split
        super().__init__(
            ann_file='',
            data_prefix=data_prefix,
            img_suffix='.pt',
            seg_map_suffix='.pt',
            ignore_index=ignore_index,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

    def load_data_list(self):
        split_dir = Path(self.data_root) / f'pastis_r_{self.split}'
        img_dir = split_dir / 's2_images'
        targets_path = split_dir / 'targets.pt'
        months_path = split_dir / 'months.pt'

        targets = torch.load(targets_path, map_location='cpu')
        months = torch.load(months_path, map_location='cpu')

        data_list = []
        for idx in range(len(targets)):
            data_list.append(
                dict(
                    img_path=str(img_dir / f'{idx}.pt'),
                    seg_map_path=str(targets_path),
                    label_map=self.label_map,
                    reduce_zero_label=self.reduce_zero_label,
                    seg_fields=[],
                    target_index=idx,
                    months=months[idx].tolist(),
                ))
        return data_list
