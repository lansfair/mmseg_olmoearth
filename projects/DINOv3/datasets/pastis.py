from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.transforms import BaseTransform
from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS, TRANSFORMS


@DATASETS.register_module()
class PASTISDataset64(BaseSegDataset):
    """PASTIS semantic segmentation dataset stored as PyTorch .pt files.

    Expected directory layout:

        pastis_dataset_64/
        ├── pastis_r_train/
        │   ├── s2_images/
        │   │   ├── 0.pt
        │   │   ├── 1.pt
        │   │   └── ...
        │   └── targets.pt
        ├── pastis_r_val/
        │   ├── s2_images/
        │   └── targets.pt
        └── pastis_r_test/
            ├── s2_images/
            └── targets.pt

    Each image file is expected to contain a tensor shaped (T, C, H, W).
    For your current preprocessed PASTIS data, this should be (12, 13, 64, 64).

    The corresponding targets.pt is expected to be shaped (N, H, W).

    Labels:
        0 to 18: valid classes
        -1: ignored label
    """

    METAINFO = dict(
        classes=tuple(f'class_{idx}' for idx in range(19)),
        palette=[
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
        ],
        ignore_index=-1,
    )

    SPLIT_DIRS = dict(
        train='pastis_r_train',
        val='pastis_r_val',
        test='pastis_r_test',
    )

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        img_dir_name: str = 's2_images',
        target_filename: str = 'targets.pt',
        ignore_index: int = -1,
        **kwargs,
    ) -> None:
        if split not in self.SPLIT_DIRS:
            raise ValueError(
                f'Unsupported split: {split}. '
                f'Expected one of {tuple(self.SPLIT_DIRS)}.'
            )

        self.split = split
        self.img_dir_name = img_dir_name
        self.target_filename = target_filename

        super().__init__(
            data_root=data_root,
            img_suffix='.pt',
            seg_map_suffix='.pt',
            reduce_zero_label=False,
            ignore_index=ignore_index,
            **kwargs,
        )

    @staticmethod
    def _sort_pt_files(paths: Sequence[Path]) -> List[Path]:
        def sort_key(path: Path):
            try:
                return (0, int(path.stem))
            except ValueError:
                return (1, path.stem)

        return sorted(paths, key=sort_key)

    def load_data_list(self) -> List[dict]:
        data_root = Path(self.data_root).expanduser().resolve()
        split_root = data_root / self.SPLIT_DIRS[self.split]
        img_dir = split_root / self.img_dir_name
        target_path = split_root / self.target_filename

        if not split_root.is_dir():
            raise FileNotFoundError(f'PASTIS split directory not found: {split_root}')

        if not img_dir.is_dir():
            raise FileNotFoundError(f'PASTIS image directory not found: {img_dir}')

        if not target_path.is_file():
            raise FileNotFoundError(f'PASTIS target file not found: {target_path}')

        targets = torch.load(str(target_path), map_location='cpu')

        if isinstance(targets, dict):
            for key in ('targets', 'target', 'labels', 'label', 'masks', 'mask'):
                if key in targets:
                    targets = targets[key]
                    break

        if not torch.is_tensor(targets):
            raise TypeError(
                f'Expected {target_path} to contain a tensor or a dict '
                'containing targets / labels / masks.'
            )

        if targets.ndim != 3:
            raise ValueError(
                f'Expected targets.pt shape to be (N, H, W), '
                f'got {tuple(targets.shape)}.'
            )

        num_targets = int(targets.shape[0])
        image_paths = self._sort_pt_files(list(img_dir.glob('*.pt')))

        if len(image_paths) == 0:
            raise FileNotFoundError(f'No .pt image files found in: {img_dir}')

        data_list = []

        for image_path in image_paths:
            try:
                sample_idx = int(image_path.stem)
            except ValueError as exc:
                raise ValueError(
                    f'Image file name must be an integer index, '
                    f'got: {image_path.name}'
                ) from exc

            if sample_idx < 0 or sample_idx >= num_targets:
                raise IndexError(
                    f'Image index {sample_idx} from {image_path.name} is outside '
                    f'target range [0, {num_targets - 1}].'
                )

            data_list.append(
                dict(
                    img_path=str(image_path),
                    seg_map_path=str(target_path),
                    sample_idx=sample_idx,
                    split=self.split,
                    reduce_zero_label=False,
                    seg_fields=[],
                )
            )

        if len(data_list) != num_targets:
            print(
                '[PASTISDataset64] Warning: number of image files '
                f'({len(data_list)}) != number of targets ({num_targets}). '
                'The dataset will use the discovered image files only.'
            )

        return data_list


@TRANSFORMS.register_module()
class LoadPastisSampleFromPT(BaseTransform):
    """Load one PASTIS image tensor and its semantic label from .pt files.

    Args:
        temporal_mode: How to reduce the temporal dimension.
            Supported values:
                mean: average 12 months into one image.
                select: select one month by time_index.
                max: temporal max.
                flatten: reshape (T, C, H, W) into (T*C, H, W).

        time_index: Month index used when temporal_mode='select'.

        band_indices: Optional channel indices after your 13-band construction.
            For RGB-like DINOv3 input from Sentinel-2, use (3, 2, 1),
            corresponding to B4 / B3 / B2 in your 13-channel order.

            Leave as None to keep all 13 bands.

        resize_size: Optional output spatial size (height, width).
            Images use bilinear interpolation.
            Labels use nearest-neighbor interpolation.

        ignore_index: Label value to ignore before optional conversion.

        target_ignore_index: If not None, convert ignore_index labels to this value.
            Keep None if you want labels to remain -1.

        to_float32: Convert image tensor to float32.
    """

    _TARGET_CACHE: Dict[str, torch.Tensor] = {}

    def __init__(
        self,
        temporal_mode: str = 'mean',
        time_index: int = 0,
        band_indices: Optional[Sequence[int]] = None,
        resize_size: Optional[Union[int, Sequence[int]]] = None,
        ignore_index: int = -1,
        target_ignore_index: Optional[int] = None,
        to_float32: bool = True,
    ) -> None:
        valid_modes = {'mean', 'select', 'max', 'flatten'}
        if temporal_mode not in valid_modes:
            raise ValueError(
                f'Unsupported temporal_mode: {temporal_mode}. '
                f'Expected one of {tuple(sorted(valid_modes))}.'
            )

        self.temporal_mode = temporal_mode
        self.time_index = int(time_index)
        self.band_indices = (
            None if band_indices is None
            else tuple(int(i) for i in band_indices)
        )
        self.resize_size = self._format_resize_size(resize_size)
        self.ignore_index = int(ignore_index)
        self.target_ignore_index = target_ignore_index
        self.to_float32 = to_float32

    @staticmethod
    def _format_resize_size(
        resize_size: Optional[Union[int, Sequence[int]]]
    ) -> Optional[Tuple[int, int]]:
        if resize_size is None:
            return None

        if isinstance(resize_size, int):
            return (resize_size, resize_size)

        if len(resize_size) != 2:
            raise ValueError('resize_size must be an int or a sequence of two ints.')

        return (int(resize_size[0]), int(resize_size[1]))

    @staticmethod
    def _extract_tensor(obj, path: str, kind: str) -> torch.Tensor:
        if isinstance(obj, dict):
            candidate_keys = [
                kind,
                f'{kind}s',
                'img',
                'image',
                's2',
                'data',
                'target',
                'targets',
                'label',
                'labels',
                'mask',
                'masks',
            ]

            for key in candidate_keys:
                if key in obj:
                    obj = obj[key]
                    break

        if not torch.is_tensor(obj):
            raise TypeError(f'Expected {path} to contain a tensor, got {type(obj)}.')

        return obj

    @classmethod
    def _load_targets(cls, path: str) -> torch.Tensor:
        if path not in cls._TARGET_CACHE:
            obj = torch.load(path, map_location='cpu')
            cls._TARGET_CACHE[path] = cls._extract_tensor(obj, path, kind='target')

        return cls._TARGET_CACHE[path]

    def _load_image(self, path: str) -> torch.Tensor:
        obj = torch.load(path, map_location='cpu')
        image = self._extract_tensor(obj, path, kind='image')

        if image.ndim != 4:
            raise ValueError(
                f'Expected image tensor shape (T, C, H, W), '
                f'got {tuple(image.shape)} from {path}.'
            )

        if image.shape[1] != 13:
            raise ValueError(
                f'Expected 13 Sentinel-2 channels, '
                f'got {image.shape[1]} from {path}.'
            )

        if self.to_float32:
            image = image.float()

        return image

    def _select_bands_4d(self, image: torch.Tensor) -> torch.Tensor:
        if self.band_indices is None:
            return image

        return image[:, list(self.band_indices), :, :]

    def _reduce_temporal(self, image: torch.Tensor) -> torch.Tensor:
        image = self._select_bands_4d(image)

        if self.temporal_mode == 'mean':
            return image.mean(dim=0)

        if self.temporal_mode == 'max':
            return image.max(dim=0).values

        if self.temporal_mode == 'select':
            num_times = int(image.shape[0])

            if self.time_index < 0:
                time_index = self.time_index + num_times
            else:
                time_index = self.time_index

            if time_index < 0 or time_index >= num_times:
                raise IndexError(
                    f'time_index={self.time_index} is outside valid range '
                    f'[-{num_times}, {num_times - 1}].'
                )

            return image[time_index]

        if self.temporal_mode == 'flatten':
            t, c, h, w = image.shape
            return image.reshape(t * c, h, w)

        raise RuntimeError(f'Unexpected temporal_mode: {self.temporal_mode}')

    @staticmethod
    def _resize_image(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        image = F.interpolate(
            image.unsqueeze(0),
            size=size,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)

        return image

    @staticmethod
    def _resize_seg(seg: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        seg = F.interpolate(
            seg[None, None].float(),
            size=size,
            mode='nearest',
        )[0, 0]

        return seg.long()

    def transform(self, results: dict) -> dict:
        image = self._load_image(results['img_path'])
        image = self._reduce_temporal(image)

        targets = self._load_targets(results['seg_map_path'])
        sample_idx = int(results['sample_idx'])

        if targets.ndim != 3:
            raise ValueError(
                f'Expected targets shape (N, H, W), '
                f'got {tuple(targets.shape)} from {results["seg_map_path"]}.'
            )

        seg = targets[sample_idx].long()

        if self.target_ignore_index is not None:
            seg = seg.clone()
            seg[seg == self.ignore_index] = int(self.target_ignore_index)

        if self.resize_size is not None:
            image = self._resize_image(image, self.resize_size)
            seg = self._resize_seg(seg, self.resize_size)

        # PackSegInputs expects image as HWC ndarray and gt_seg_map as HW ndarray.
        image_np = image.permute(1, 2, 0).contiguous().cpu().numpy()
        seg_np = seg.contiguous().cpu().numpy().astype(np.int64)

        h, w = int(image_np.shape[0]), int(image_np.shape[1])

        results['img'] = image_np
        results['gt_seg_map'] = seg_np
        results['ori_shape'] = (h, w)
        results['img_shape'] = (h, w)
        results['pad_shape'] = (h, w)
        results['scale_factor'] = 1.0
        results['seg_fields'] = ['gt_seg_map']
        results['num_channels'] = int(image_np.shape[2])

        return results
