from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from mmengine.dataset import BaseDataset


def _build_palette(num_classes: int = 19) -> List[List[int]]:
    """Build a deterministic generic palette.

    Replace this palette later if you have the official PASTIS class colors.
    """
    palette = []
    for i in range(num_classes):
        palette.append([
            (37 * i + 17) % 256,
            (67 * i + 29) % 256,
            (97 * i + 43) % 256,
        ])
    return palette


def _numeric_sort_key(path: Path):
    try:
        return int(path.stem)
    except ValueError:
        return path.stem


def _torch_load(path: Path):
    """torch.load wrapper compatible with different PyTorch versions."""
    try:
        return torch.load(str(path), map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location='cpu')


def _collect_registries():
    registries = []
    try:
        from mmseg.registry import DATASETS as MMSEG_DATASETS
        registries.append(MMSEG_DATASETS)
    except Exception:
        pass

    # Extra registration to the root MMEngine registry makes the class more
    # tolerant when default_scope is not set correctly.
    try:
        from mmengine.registry import DATASETS as MMENGINE_DATASETS
        registries.append(MMENGINE_DATASETS)
    except Exception:
        pass

    unique = []
    seen = set()
    for registry in registries:
        if id(registry) not in seen:
            unique.append(registry)
            seen.add(id(registry))
    return unique


def _register_dataset(cls):
    for registry in _collect_registries():
        registry.register_module(module=cls, force=True)
    return cls


_CLASSES = tuple(f'class_{i}' for i in range(19))


@_register_dataset
class PastisPtDataset(BaseDataset):
    """PASTIS semantic segmentation dataset stored as .pt tensors.

    Expected data layout:

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

    Each image file is expected to be shaped as:
        T x C x H x W, normally 12 x 13 x 64 x 64.

    Each targets.pt is expected to be shaped as:
        N x H x W.

    Labels:
        0..18 are valid classes.
        -1 is ignored in the original target file.
    """

    METAINFO = dict(
        classes=_CLASSES,
        palette=_build_palette(len(_CLASSES)),
    )

    def __init__(
        self,
        data_root: str,
        split: str,
        pipeline: Sequence[Dict[str, Any]],
        image_dir: str = 's2_images',
        targets_file: str = 'targets.pt',
        file_suffix: str = '.pt',
        target_index_from_filename: bool = True,
        validate_files: bool = True,
        metainfo: Optional[Dict[str, Any]] = None,
        serialize_data: bool = False,
        **kwargs,
    ) -> None:
        self.split = split
        self.image_dir = image_dir
        self.targets_file = targets_file
        self.file_suffix = file_suffix
        self.target_index_from_filename = target_index_from_filename
        self.validate_files = validate_files

        super().__init__(
            ann_file='',
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=dict(),
            pipeline=pipeline,
            serialize_data=serialize_data,
            **kwargs,
        )

    def _resolve_targets_path(self, split_dir: Path) -> Path:
        targets_path = split_dir / self.targets_file

        # Tolerate an accidentally nested targets.pt, because dataset sketches
        # sometimes place it under s2_images by indentation mistake.
        if not targets_path.is_file():
            alt_path = split_dir / self.image_dir / self.targets_file
            if alt_path.is_file():
                targets_path = alt_path

        if not targets_path.is_file():
            raise FileNotFoundError(
                f'Cannot find targets file. Tried: {split_dir / self.targets_file} '
                f'and {split_dir / self.image_dir / self.targets_file}'
            )
        return targets_path

    def load_data_list(self) -> List[Dict[str, Any]]:
        split_dir = Path(self.data_root) / self.split
        img_dir = split_dir / self.image_dir

        if not split_dir.is_dir():
            raise FileNotFoundError(f'Split directory not found: {split_dir}')
        if not img_dir.is_dir():
            raise FileNotFoundError(f'Image directory not found: {img_dir}')

        image_paths = sorted(img_dir.glob(f'*{self.file_suffix}'), key=_numeric_sort_key)
        if len(image_paths) == 0:
            raise FileNotFoundError(
                f'No image files with suffix {self.file_suffix!r} found in {img_dir}'
            )

        targets_path = self._resolve_targets_path(split_dir)

        target_count = None
        if self.validate_files:
            targets = _torch_load(targets_path)
            if isinstance(targets, dict):
                # Common fallback if someone saved {'targets': tensor}.
                for key in ('targets', 'target', 'mask', 'masks', 'labels', 'label'):
                    if key in targets:
                        targets = targets[key]
                        break
            if not torch.is_tensor(targets):
                raise TypeError(f'targets.pt must contain a Tensor or a dict containing a Tensor: {targets_path}')
            if targets.ndim not in (3, 4):
                raise ValueError(
                    f'targets.pt should have shape N x H x W or N x 1 x H x W, '
                    f'but got shape {tuple(targets.shape)} from {targets_path}'
                )
            target_count = int(targets.shape[0])

        data_list = []
        for ordinal, img_path in enumerate(image_paths):
            if self.target_index_from_filename:
                try:
                    target_index = int(img_path.stem)
                except ValueError as exc:
                    raise ValueError(
                        f'target_index_from_filename=True requires numeric image names, '
                        f'but got {img_path.name}'
                    ) from exc
            else:
                target_index = ordinal

            if target_count is not None and target_index >= target_count:
                raise IndexError(
                    f'Image {img_path.name} maps to target index {target_index}, '
                    f'but {targets_path} contains only {target_count} targets.'
                )

            data_list.append(dict(
                img_path=str(img_path),
                targets_path=str(targets_path),
                sample_idx=ordinal,
                target_index=target_index,
                img_id=img_path.stem,
                split=self.split,
            ))

        return data_list
