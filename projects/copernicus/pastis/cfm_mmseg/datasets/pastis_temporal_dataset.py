import copy
import re
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from mmengine.dataset import BaseDataset
from mmseg.registry import DATASETS


PASTIS_CLASSES = (
    'background',
    'class_1', 'class_2', 'class_3', 'class_4', 'class_5',
    'class_6', 'class_7', 'class_8', 'class_9', 'class_10',
    'class_11', 'class_12', 'class_13', 'class_14', 'class_15',
    'class_16', 'class_17', 'class_18',
)

PASTIS_PALETTE = [
    [0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25],
    [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240],
    [240, 50, 230], [210, 245, 60], [250, 190, 190], [0, 128, 128],
    [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
    [170, 255, 195], [128, 128, 0], [255, 215, 180],
]


def _safe_torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _natural_key(path: Path):
    stem = path.stem
    parts = re.split(r'(\d+)', stem)
    return [int(p) if p.isdigit() else p for p in parts]


@DATASETS.register_module(force=True)
class PASTISTemporalPtDataset(BaseDataset):
    """Dataset for the processed PASTIS-R temporal segmentation format.

    Expected directory structure::

        data_root/
          pastis_r_train/
            s2_images/0.pt
            s2_images/1.pt
            months.pt          # optional
            targets.pt
          pastis_r_valid/ or pastis_r_val/
          pastis_r_test/

    Each image tensor is T x C x H x W. For your processor, it is usually
    12 x 13 x 128 x 128 or 12 x 13 x 64 x 64.
    """

    METAINFO = dict(classes=PASTIS_CLASSES, palette=PASTIS_PALETTE)

    def __init__(
        self,
        data_root: str,
        split: str,
        pipeline: Sequence,
        img_dir: str = 's2_images',
        target_file: str = 'targets.pt',
        months_file: str = 'months.pt',
        img_suffix: str = '.pt',
        metainfo: Optional[dict] = None,
        serialize_data: bool = False,
        lazy_init: bool = False,
        max_refetch: int = 1000,
    ):
        self.split = split
        self.img_dir = img_dir
        self.target_file = target_file
        self.months_file = months_file
        self.img_suffix = img_suffix
        self.targets = None
        super().__init__(
            data_root=data_root,
            metainfo=metainfo,
            pipeline=pipeline,
            serialize_data=serialize_data,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
        )

    def _split_dir(self) -> Path:
        root = Path(self.data_root)
        if not self.split:
            return root
        if root.name == self.split:
            return root
        return root / self.split

    def load_data_list(self) -> List[dict]:
        split_dir = self._split_dir()
        image_dir = split_dir / self.img_dir
        target_path = split_dir / self.target_file
        months_path = split_dir / self.months_file

        if not image_dir.is_dir():
            raise FileNotFoundError(
                f'Image directory not found: {image_dir}. Check data_root={self.data_root!r}, split={self.split!r}.'
            )
        if not target_path.is_file():
            raise FileNotFoundError(f'Target file not found: {target_path}.')

        img_files = sorted(image_dir.glob(f'*{self.img_suffix}'), key=_natural_key)
        if len(img_files) == 0:
            raise RuntimeError(f'No image files found in {image_dir} with suffix {self.img_suffix!r}.')

        targets = _safe_torch_load(target_path)
        if isinstance(targets, dict):
            for key in ('targets', 'target', 'labels', 'masks', 'y'):
                if key in targets:
                    targets = targets[key]
                    break
        if not torch.is_tensor(targets):
            targets = torch.as_tensor(targets)
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets[:, 0]
        if targets.ndim != 3:
            raise ValueError(f'targets.pt should have shape N x H x W, got {tuple(targets.shape)}')
        if targets.shape[0] != len(img_files):
            raise ValueError(f'Number of images and targets mismatch: {len(img_files)} images vs {targets.shape[0]} targets.')
        self.targets = targets.long()

        data_list = []
        for i, img_path in enumerate(img_files):
            info = dict(
                img_path=str(img_path),
                target_path=str(target_path),
                target_index=i,
                sample_idx=i,
                seg_fields=[],
            )
            if months_path.is_file():
                info['months_path'] = str(months_path)
            data_list.append(info)
        return data_list

    def prepare_data(self, idx):
        if not self._fully_initialized:
            self.full_init()
        data_info = copy.deepcopy(self.get_data_info(idx))
        data_info['gt_seg_map'] = self.targets[idx].clone()
        return self.pipeline(data_info)


# Also expose the dataset in MMEngine's root dataset registry for helper scripts
# or environments that build datasets without the mmseg scope.
try:  # pragma: no cover
    from mmengine.registry import DATASETS as MMENGINE_DATASETS
    if MMENGINE_DATASETS.get('PASTISTemporalPtDataset') is None:
        MMENGINE_DATASETS.register_module(module=PASTISTemporalPtDataset, force=True)
except Exception:
    pass
