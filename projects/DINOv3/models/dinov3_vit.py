import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS


@MODELS.register_module()
class DINOv3ViT(BaseModule):
    """DINOv3 ViT backbone wrapper for MMSegmentation.

    The wrapper imports the official DINOv3 source code, builds a DINOv3 ViT
    backbone, loads a local checkpoint, and returns intermediate patch-token
    feature maps in MMSegmentation format.

    It is intentionally generic. You can switch between different DINOv3 ViT
    variants by changing ``model_name``, ``weights_name``, ``weights_path`` and
    ``out_indices`` in your MMSegmentation config.

    Expected official DINOv3 source layout when ``repo_path`` is not specified:

    .. code-block:: text

        mmsegmentation/
        └── projects/
            └── DINOv3/
                ├── models/
                │   └── dinov3_vit.py
                └── dinov3/
                    ├── hubconf.py
                    └── dinov3/
                        └── hub/
                            └── backbones.py

    Args:
        model_name (str): Function name in ``dinov3.hub.backbones``.
            Examples: ``dinov3_vits16``, ``dinov3_vitb16``,
            ``dinov3_vitl16``, ``dinov3_vitl16plus``.
        repo_path (str, optional): Official DINOv3 repository root. The path
            should contain the lower-case ``dinov3`` package. If omitted, this
            wrapper tries to find ``projects/DINOv3/dinov3`` automatically.
        weights_path (str, optional): Local DINOv3 checkpoint path.
        weights_name (str, optional): Official DINOv3 weight enum name, such as
            ``LVD1689M`` or ``SAT493M``. For the satellite-trained ViT-L/16
            checkpoint, use ``SAT493M``.
        out_indices (Sequence[int], optional): Zero-based transformer block
            indices used as output feature stages. If omitted, four indices are
            selected automatically according to model depth. For ViT-L/16 depth
            24 this becomes ``(5, 11, 17, 23)``; for ViT-S/16 depth 12 this
            becomes ``(2, 5, 8, 11)``.
        patch_size (int): Patch size, usually 16 for DINOv3 ViT.
        load_strict (bool): Whether to strictly load ``weights_path``.
        freeze (bool): Whether to freeze all DINOv3 backbone parameters.
        norm (bool): Whether to apply DINOv3 final norm to intermediate tokens.
        model_kwargs (dict, optional): Extra keyword arguments passed to the
            official DINOv3 builder.
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
        self.patch_size = patch_size
        self.load_strict = load_strict
        self.freeze = freeze
        self.norm = norm
        self.model_kwargs = model_kwargs or {}

        self.dinov3 = self._build_dinov3_model()
        self.patch_size = int(getattr(self.dinov3, 'patch_size', self.patch_size))

        self.depth = self._infer_depth()
        self.out_indices = self._format_out_indices(out_indices, self.depth)
        self.out_channels = self._infer_out_channels()
        self.num_features = self.out_channels

        if self.weights_path:
            self.load_dinov3_weights(self.weights_path, strict=self.load_strict)

        if self.freeze:
            self._freeze_backbone()

    def _candidate_repo_paths(self) -> list:
        paths = []

        if self.repo_path:
            paths.append(Path(self.repo_path).expanduser())

        env_path = os.environ.get('DINOV3_REPO_PATH')
        if env_path:
            paths.append(Path(env_path).expanduser())

        project_root = Path(__file__).resolve().parents[1]
        paths.extend([
            project_root / 'dinov3',
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
        for path in self._candidate_repo_paths():
            path = path.resolve()
            if self._is_dinov3_repo(path):
                path_str = str(path)
                if path_str not in sys.path:
                    sys.path.insert(0, path_str)
                return

        searched = '\n'.join(str(p) for p in self._candidate_repo_paths())
        raise FileNotFoundError(
            'Cannot find the official DINOv3 source repository. '
            'Please set `repo_path` in the backbone config or set the '
            'DINOV3_REPO_PATH environment variable. Searched paths:\n'
            f'{searched}'
        )

    def _build_dinov3_model(self) -> nn.Module:
        self._prepare_dinov3_import()

        try:
            from dinov3.hub import backbones as dinov3_backbones
            from dinov3.hub.backbones import Weights
        except Exception as exc:
            raise ImportError(
                'Failed to import DINOv3. Make sure the official DINOv3 '
                'repository root is available and contains `dinov3/hub`.'
            ) from exc

        if not hasattr(dinov3_backbones, self.model_name):
            available = [
                name for name in dir(dinov3_backbones)
                if name.startswith('dinov3_')
            ]
            raise ValueError(
                f'Unsupported DINOv3 model_name: {self.model_name}. '
                f'Available candidates: {available}'
            )

        build_fn = getattr(dinov3_backbones, self.model_name)
        kwargs = dict(self.model_kwargs)

        weights = None
        if self.weights_name:
            if hasattr(Weights, self.weights_name):
                weights = getattr(Weights, self.weights_name)
            else:
                weights = self.weights_name

        # Important:
        # Do not use pretrained=True here. For local checkpoint names without
        # official hash suffixes, DINOv3's hub loader may reject the filename or
        # try to download from URL. We build the architecture first, then load
        # the local checkpoint ourselves.
        if weights is None:
            model = build_fn(pretrained=False, **kwargs)
        else:
            model = build_fn(pretrained=False, weights=weights, **kwargs)

        return model

    def _infer_depth(self) -> int:
        blocks = getattr(self.dinov3, 'blocks', None)
        if blocks is None:
            raise AttributeError('The DINOv3 model does not have `blocks`.')
        return len(blocks)

    def _format_out_indices(
        self,
        out_indices: Optional[Sequence[int]],
        depth: int,
    ) -> Tuple[int, ...]:
        if out_indices is None:
            out_indices = (
                depth // 4 - 1,
                depth // 2 - 1,
                depth * 3 // 4 - 1,
                depth - 1,
            )

        formatted = []
        for index in out_indices:
            index = int(index)
            if index < 0:
                index += depth
            if index < 0 or index >= depth:
                raise ValueError(
                    f'Invalid out index {index} for DINOv3 depth {depth}.'
                )
            formatted.append(index)

        return tuple(formatted)

    def _infer_out_channels(self) -> int:
        if hasattr(self.dinov3, 'embed_dim'):
            return int(self.dinov3.embed_dim)

        norm = getattr(self.dinov3, 'norm', None)
        if norm is not None and hasattr(norm, 'normalized_shape'):
            return int(norm.normalized_shape[0])

        raise AttributeError(
            'Cannot infer DINOv3 output channels. Please check the official '
            'DINOv3 model implementation.'
        )

    @staticmethod
    def _extract_state_dict(checkpoint):
        if isinstance(checkpoint, dict):
            for key in ('state_dict', 'model', 'teacher', 'student'):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    checkpoint = value
                    break

        if not isinstance(checkpoint, dict):
            raise TypeError(
                'Unsupported checkpoint format. Expected a state_dict or a '
                'dict containing one of: state_dict, model, teacher, student.'
            )

        prefixes = (
            'module.',
            'backbone.',
            'model.',
            'teacher.',
            'student.',
            'encoder.',
        )

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
        weights_path = Path(weights_path).expanduser().resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f'DINOv3 checkpoint not found: {weights_path}')

        checkpoint = torch.load(str(weights_path), map_location='cpu')
        state_dict = self._extract_state_dict(checkpoint)
        result = self.dinov3.load_state_dict(state_dict, strict=strict)

        if strict:
            print(f'[DINOv3ViT] Loaded checkpoint with strict=True: {weights_path}')
        else:
            missing = getattr(result, 'missing_keys', [])
            unexpected = getattr(result, 'unexpected_keys', [])
            print(
                f'[DINOv3ViT] Loaded checkpoint with strict=False: {weights_path}; '
                f'missing={len(missing)}, unexpected={len(unexpected)}'
            )

    def _freeze_backbone(self) -> None:
        self.dinov3.eval()
        for param in self.dinov3.parameters():
            param.requires_grad = False

    def init_weights(self) -> None:
        # The official DINOv3 builder initializes the model. Local checkpoint
        # loading is handled in __init__ by load_dinov3_weights().
        pass

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self._freeze_backbone()
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        height, width = x.shape[-2:]
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f'Input size {(height, width)} must be divisible by '
                f'patch_size={self.patch_size}. Please use crop/pad sizes '
                'that are multiples of the DINOv3 patch size.'
            )

        outs = self.dinov3.get_intermediate_layers(
            x,
            n=self.out_indices,
            reshape=True,
            return_class_token=False,
            return_extra_tokens=False,
            norm=self.norm,
        )

        return tuple(outs)
