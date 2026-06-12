import os
from pathlib import Path
from typing import List
from typing import Dict

import rasterio
import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.datasets import BaseSegDataset
from mmseg.registry import TRANSFORMS
from mmseg.registry import DATASETS


geobench_dataset = None


@TRANSFORMS.register_module(force=True)
class LoadSingleRSImgFromHDF5(BaseTransform):
    def __init__(self, bands, convert_to_u16=False):
        self.bands = bands
        self.convert_to_u16 = convert_to_u16

    def transform(self, results: Dict) -> Dict:
        if geobench_dataset is not None:
            sample = geobench_dataset.get_sample(results['sample_name'])
            img = sample.pack_to_3d(band_names=self.bands)[0]
        else:
            with rasterio.open(results['img_path']) as src:
                img = src.read()
            if img.ndim == 3:
                img = np.moveaxis(img, 0, 2)
        if self.convert_to_u16 and img.dtype is not np.uint16:
            img = img.astype(np.uint16)
        results['img'] = img
        results['img_shape'] = img.shape[: 2]
        results['ori_shape'] = img.shape[: 2]
        return results
    

@TRANSFORMS.register_module(force=True)
class LoadSingleRSAnnFromHDF5(BaseTransform):
    def transform(self, results: Dict) -> Dict:
        if geobench_dataset is None:
            return results
        sample = geobench_dataset.get_sample(results['sample_name'])
        gt_semantic_seg = sample.label.data.astype('uint8')
        if results['reduce_zero_label']:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')
        return results


class GEOBenchSegDataset(BaseSegDataset):
    def __init__(self, data_root: str, split: str = 'train', **kwargs) -> None:
        data_root = Path(data_root)
        os.environ['GEO_BENCH_DIR'] = data_root.parent.parent.as_posix()
        import geobench
        task = geobench.load_task_specs(data_root)
        global geobench_dataset
        geobench_dataset = task.get_dataset(split=split, band_names=None)
        super().__init__(data_root=data_root.as_posix(), **kwargs)

    def load_data_list(self) -> List[dict]:
        data_list = []
        sample_name_list = geobench_dataset.active_partition.partition_dict[geobench_dataset.split]
        for sample_name in sample_name_list:
            data_info = dict(
                sample_name=sample_name,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields = []
            )
            data_list.append(data_info)
        return data_list
    

@DATASETS.register_module(force=True)
class CashewPlantSegDataset(GEOBenchSegDataset):
    METAINFO = {
        'classes': [
            '0',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6'
        ],
        'palette': [
            [255, 255, 255], 
            [255, 0, 0], 
            [255, 255, 0], 
            [0, 0, 255],
            [159, 129, 183], 
            [0, 255, 0], 
            [255, 195, 128]
        ]
    }


@DATASETS.register_module(force=True)
class ChesapeakeSegDataset(GEOBenchSegDataset):
    METAINFO = {
        'classes': [
            '0',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6'
        ],
        'palette': [
            [255, 255, 255], 
            [255, 0, 0], 
            [255, 255, 0], 
            [0, 0, 255],
            [159, 129, 183], 
            [0, 255, 0], 
            [255, 195, 128]
        ]
    }


@DATASETS.register_module(force=True)
class NeonTreeSegDataset(GEOBenchSegDataset):
    METAINFO = {
        'classes': [
            '0',
            '1'
        ],
        'palette': [
            [255, 255, 255], 
            [255, 0, 0]
        ]
    }


@DATASETS.register_module(force=True)
class NZCattleSegDataset(GEOBenchSegDataset):
    METAINFO = {
        'classes': [
            '0',
            '1'
        ],
        'palette': [
            [255, 255, 255], 
            [255, 0, 0]
        ]
    }


@DATASETS.register_module(force=True)
class Pv4gerSegDataset(GEOBenchSegDataset):
    METAINFO = {
        'classes': [
            '0',
            '1'
        ],
        'palette': [
            [255, 255, 255], 
            [255, 0, 0]
        ]
    }


@DATASETS.register_module(force=True)
class SACropTypeSegDataset(GEOBenchSegDataset):
    METAINFO = {
        'classes': [
            '0',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9'
        ],
        'palette': [
            [0, 0, 0], 
            [128, 0, 0], 
            [0, 128, 0], 
            [128, 128, 0],
            [0, 0, 128], 
            [128, 0, 128], 
            [0, 128, 128], 
            [128, 128, 128],
            [64, 0, 0], 
            [192, 0, 0]
        ]
    }
