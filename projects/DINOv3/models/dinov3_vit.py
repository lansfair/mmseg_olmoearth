from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS


@MODELS.register_module()
class DINOv3ViT(BaseModule):
    """Official DINOv3 ViT wrapper for MMSegmentation.

    The architecture is built from the official source under
    ``projects/DINOv3/dinov3`` and a local checkpoint is loaded afterwards.
    Passing ``weights_name='SAT493M'`` is important for ViT-L/16 because that
    checkpoint uses the SAT493M-specific architecture flag.
    """

    def __init__(
        self,
        model_name: str = 'dinov3_vitl16',
        repo_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        weights_name: Optional[str] = 'SAT493M',
        out_indices: Optional[Sequence[int]] = None,
        patch_size: int = 16,
        load_strict: bool = True,
        freeze: bool = False,
        norm: bool = True,
        model_kwargs: Optional[dict] = None,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.model_name = model_name
        self.repo_path = repo_path
        self.weights_path = weights_path
        self.weights_name = weights_name
        self.patch_size = int(patch_size)
        self.load_strict = bool(load_strict)
        self.freeze = bool(freeze)
        self.norm = bool(norm)
        self.model_kwargs = model_kwargs or {}

        self.dinov3 = self._build_dinov3_model()
        self.patch_size = int(getattr(self.dinov3, 'patch_size', self.patch_size))
        self.depth = self._infer_depth()
        self.out_indices = self._format_out_indices(out_indices, self.depth)
        self.out_channels = self._infer_out_channels()
        self.num_features = self.out_channels

        if self.weights_path:
            self.load_dinov3_weights(self.weights_path, strict=self.load_strict)
        self.set_frozen(self.freeze)

    def _candidate_repo_paths(self) -> list[Path]:
        paths: list[Path] = []
        if self.repo_path:
            paths.append(Path(self.repo_path).expanduser())
        if os.environ.get('DINOV3_REPO_PATH'):
            paths.append(Path(os.environ['DINOV3_REPO_PATH']).expanduser())

        project_root = Path(__file__).resolve().parents[1]
        paths.extend([
            project_root,
            project_root / 'dinov3-main',
            project_root / 'third_party' / 'dinov3',
            project_root / 'third_party' / 'dinov3-main',
            project_root.parent.parent / 'third_party' / 'dinov3',
            project_root.parent.parent / 'third_party' / 'dinov3-main',
        ])
        return paths

    @staticmethod
    def _is_dinov3_repo(path: Path) -> bool:
        return (path / 'dinov3' / 'hub' / 'backbones.py').is_file()

    def _prepare_dinov3_import(self) -> None:
        searched = []
        for path in self._candidate_repo_paths():
            path = path.resolve()
            searched.append(str(path))
            if self._is_dinov3_repo(path):
                path_str = str(path)
                if path_str not in sys.path:
                    sys.path.insert(0, path_str)
                return
        raise FileNotFoundError(
            'Cannot find official DINOv3 source. Set repo_path or '
            'DINOV3_REPO_PATH. Searched:\n' + '\n'.join(searched)
        )

    def _build_dinov3_model(self) -> nn.Module:
        self._prepare_dinov3_import()
        try:
            from dinov3.hub import backbones as dinov3_backbones
            from dinov3.hub.backbones import Weights
        except Exception as exc:
            raise ImportError(
                'Failed to import official DINOv3 from its repository root.'
            ) from exc

        if not hasattr(dinov3_backbones, self.model_name):
            available = sorted(
                name for name in dir(dinov3_backbones)
                if name.startswith('dinov3_')
            )
            raise ValueError(
                f'Unsupported DINOv3 model_name={self.model_name!r}. '
                f'Available: {available}'
            )

        build_fn = getattr(dinov3_backbones, self.model_name)
        weights = None
        if self.weights_name:
            weights = getattr(Weights, self.weights_name, self.weights_name)
        kwargs = dict(self.model_kwargs)
        if weights is None:
            return build_fn(pretrained=False, **kwargs)
        return build_fn(pretrained=False, weights=weights, **kwargs)

    def _infer_depth(self) -> int:
        blocks = getattr(self.dinov3, 'blocks', None)
        if blocks is None:
            raise AttributeError('The DINOv3 model has no `blocks` attribute.')
        return len(blocks)

    @staticmethod
    def _format_out_indices(
        out_indices: Optional[Sequence[int]], depth: int
    ) -> Tuple[int, ...]:
        if out_indices is None:
            out_indices = (depth - 1,)
        result = []
        for index in out_indices:
            index = int(index)
            if index < 0:
                index += depth
            if not 0 <= index < depth:
                raise ValueError(f'Invalid out index {index} for depth {depth}.')
            result.append(index)
        return tuple(result)

    def _infer_out_channels(self) -> int:
        if hasattr(self.dinov3, 'embed_dim'):
            return int(self.dinov3.embed_dim)
        norm = getattr(self.dinov3, 'norm', None)
        if norm is not None and hasattr(norm, 'normalized_shape'):
            return int(norm.normalized_shape[0])
        raise AttributeError('Cannot infer DINOv3 output channels.')

    @staticmethod
    def _extract_state_dict(checkpoint) -> dict:
        if isinstance(checkpoint, dict):
            for key in ('state_dict', 'model', 'teacher', 'student'):
                if isinstance(checkpoint.get(key), dict):
                    checkpoint = checkpoint[key]
                    break
        if not isinstance(checkpoint, dict):
            raise TypeError('Checkpoint must be a state_dict or contain one.')

        prefixes = ('module.', 'backbone.', 'model.', 'teacher.', 'student.', 'encoder.')
        state_dict = {}
        for key, value in checkpoint.items():
            new_key = key
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        changed = True
            state_dict[new_key] = value
        return state_dict

    def load_dinov3_weights(self, weights_path: str, strict: bool = True) -> None:
        path = Path(weights_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'DINOv3 checkpoint not found: {path}')
        try:
            checkpoint = torch.load(str(path), map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(str(path), map_location='cpu')
        state_dict = self._extract_state_dict(checkpoint)
        result = self.dinov3.load_state_dict(state_dict, strict=strict)
        if strict:
            print(f'[DINOv3ViT] Strictly loaded: {path}')
        else:
            print(
                f'[DINOv3ViT] Loaded: {path}; '
                f'missing={len(result.missing_keys)}, '
                f'unexpected={len(result.unexpected_keys)}'
            )

    def set_frozen(self, frozen: bool) -> None:
        """Freeze/unfreeze the DINOv3 parameters without blocking input grads."""
        self.freeze = bool(frozen)
        for parameter in self.dinov3.parameters():
            parameter.requires_grad = not self.freeze
        if self.freeze:
            self.dinov3.eval()
        else:
            self.dinov3.train(self.training)

    def init_weights(self) -> None:
        # The official builder initializes the architecture; local weights were
        # already loaded in __init__ and must not be overwritten by MMEngine.
        return

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.dinov3.eval()
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                'DINOv3ViT expects (B, 3, H, W), got '
                f'{tuple(x.shape)}.'
            )
        height, width = x.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f'Input {(height, width)} must be divisible by '
                f'patch_size={self.patch_size}.'
            )
        outs = self.dinov3.get_intermediate_layers(
            x,
            n=self.out_indices,
            reshape=True,
            return_class_token=False,
            return_extra_tokens=False,
            norm=self.norm,
        )
        return tuple(out.contiguous() for out in outs)
