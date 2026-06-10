from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import torch
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


def _torch_load(path: Path):
    try:
        return torch.load(str(path), map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location='cpu')


def _extract_tensor(value, keys: Sequence[str]) -> torch.Tensor:
    if isinstance(value, dict):
        for key in keys:
            if torch.is_tensor(value.get(key)):
                value = value[key]
                break
    if not torch.is_tensor(value):
        raise TypeError('Expected a tensor or a dict containing a tensor.')
    return value


@DATASETS.register_module()
class PastisPTDataset(BaseSegDataset):
    """PASTIS split stored as ``s2_images/*.pt`` and one ``targets.pt``.

    Expected sample shape: ``(12, 13, H, W)``. Target shape: ``(N, H, W)``.
    File ``s2_images/<index>.pt`` is paired with ``targets[index]``.
    """

    METAINFO = dict(
        classes=tuple(f'class_{index}' for index in range(19)),
        palette=[
            [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
            [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
            [0, 64, 0], [128, 64, 0], [0, 192, 0],
        ],
    )

    def __init__(
        self,
        data_root: str,
        split: str,
        img_dir_name: str = 's2_images',
        target_filename: str = 'targets.pt',
        **kwargs,
    ) -> None:
        self.split_dir = str(split)
        self.img_dir_name = str(img_dir_name)
        self.target_filename = str(target_filename)
        super().__init__(
            data_root=data_root,
            img_suffix='.pt',
            seg_map_suffix='.pt',
            reduce_zero_label=False,
            **kwargs,
        )

    @staticmethod
    def _sort_key(path: Path):
        try:
            return (0, int(path.stem))
        except ValueError:
            return (1, path.stem)

    def load_data_list(self) -> List[dict]:
        split_root = Path(self.data_root).expanduser().resolve() / self.split_dir
        image_root = split_root / self.img_dir_name
        targets_path = split_root / self.target_filename
        if not image_root.is_dir():
            raise FileNotFoundError(f'Image directory not found: {image_root}')
        if not targets_path.is_file():
            raise FileNotFoundError(f'Target file not found: {targets_path}')

        targets = _extract_tensor(
            _torch_load(targets_path),
            ('targets', 'target', 'labels', 'label', 'masks', 'mask'),
        )
        if targets.ndim != 3:
            raise ValueError(f'Expected targets (N,H,W), got {tuple(targets.shape)}.')
        num_targets = int(targets.shape[0])

        image_paths = sorted(image_root.glob('*.pt'), key=self._sort_key)
        if not image_paths:
            raise FileNotFoundError(f'No .pt images found in: {image_root}')

        data_list = []
        for image_path in image_paths:
            try:
                index = int(image_path.stem)
            except ValueError as exc:
                raise ValueError(f'Image filename must be an integer: {image_path.name}') from exc
            if not 0 <= index < num_targets:
                raise IndexError(
                    f'Image index {index} is outside targets range [0, {num_targets - 1}].'
                )
            data_list.append(dict(
                img_path=str(image_path),
                targets_path=str(targets_path),
                target_index=index,
                img_id=image_path.stem,
                split=self.split_dir,
                seg_fields=[],
                reduce_zero_label=False,
            ))
        return data_list
