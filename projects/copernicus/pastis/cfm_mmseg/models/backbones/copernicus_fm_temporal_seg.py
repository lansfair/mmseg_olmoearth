from typing import List, Optional, Sequence, Tuple, Union
import math
from cfm_mmseg.copernicus_fm.flexivit.utils import resize_abs_pos_embed
import torch
from mmengine.dist import master_only
from mmengine.model import BaseModule
from mmseg.registry import MODELS

from functools import partial

import torch.nn as nn
from cfm_mmseg.copernicus_fm.models_dwv_seg import CopernicusFMViT


# Copernicus-FM Sentinel-2 13-band order:
# [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12]
S2_13_WAVELENGTHS_NM = [440, 490, 560, 665, 705, 740, 783, 842, 860, 940, 1370, 1610, 2190]
S2_13_BANDWIDTHS_NM = [20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 30, 90, 180]

ARCH_SETTINGS = {
    'small': dict(embed_dim=384, depth=12, num_heads=6, patch_size=16),
    'base': dict(embed_dim=768, depth=12, num_heads=12, patch_size=16),
    'large': dict(embed_dim=1024, depth=24, num_heads=16, patch_size=16),
    'huge': dict(embed_dim=1280, depth=32, num_heads=16, patch_size=14),
}
ARCH_OUT_INDICES = {
    'small': (3, 5, 7, 11),
    'base': (3, 5, 7, 11),
    'large': (7, 11, 15, 23),
    'huge': (7, 15, 23, 31),
}
ARCH_CHANNELS = {
    'small': 384,
    'base': 768,
    'large': 1024,
    'huge': 1280,
}


def _load_checkpoint_state(path: str):
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict):
        for key in ('model', 'state_dict', 'model_state_dict', 'net', 'network'):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise TypeError(f'Unsupported checkpoint type from {path}: {type(ckpt)}')
    state_dict = {}
    for k, v in ckpt.items():
        for prefix in ('module.', 'backbone.', 'cfm.', 'encoder.'):
            if k.startswith(prefix):
                k = k[len(prefix):]
        # If the checkpoint was saved from this wrapper, remove wrapper prefix too.
        if k.startswith('backbone.cfm.'):
            k = k[len('backbone.cfm.'):]
        state_dict[k] = v
    return state_dict
    

def _resize_pos_embed_for_current_model(state_dict: dict, model: nn.Module, key: str = 'pos_embed'):
    """Resize checkpoint absolute position embedding to current model img_size.

    Example:
        checkpoint 224x224: pos_embed [1, 197, 768]
        current    128x128: pos_embed [1, 65, 768]
        current    512x512: pos_embed [1, 1025, 768]
    """
    if key not in state_dict or not hasattr(model, key):
        return

    ckpt_pos = state_dict[key]
    model_pos = getattr(model, key)

    if ckpt_pos.shape == model_pos.shape:
        return

    if ckpt_pos.ndim != 3 or model_pos.ndim != 3:
        print(f'[CopernicusFMTemporalSegBackbone] Drop {key}: unsupported ndim '
              f'{tuple(ckpt_pos.shape)} -> {tuple(model_pos.shape)}')
        state_dict.pop(key)
        return

    if ckpt_pos.shape[0] != model_pos.shape[0] or ckpt_pos.shape[-1] != model_pos.shape[-1]:
        print(f'[CopernicusFMTemporalSegBackbone] Drop {key}: incompatible shape '
              f'{tuple(ckpt_pos.shape)} -> {tuple(model_pos.shape)}')
        state_dict.pop(key)
        return

    num_prefix_tokens = 1  # cls token

    old_tokens = ckpt_pos.shape[1] - num_prefix_tokens
    new_tokens = model_pos.shape[1] - num_prefix_tokens

    old_size = int(math.sqrt(old_tokens))
    new_size = int(math.sqrt(new_tokens))

    if old_size * old_size != old_tokens or new_size * new_size != new_tokens:
        print(f'[CopernicusFMTemporalSegBackbone] Drop {key}: non-square token count '
              f'{tuple(ckpt_pos.shape)} -> {tuple(model_pos.shape)}')
        state_dict.pop(key)
        return

    state_dict[key] = resize_abs_pos_embed(
        ckpt_pos,
        new_size=(new_size, new_size),
        old_size=(old_size, old_size),
        num_prefix_tokens=num_prefix_tokens,
    )

    print(f'[CopernicusFMTemporalSegBackbone] Resized {key}: '
          f'{old_size}x{old_size} -> {new_size}x{new_size}, '
          f'{tuple(ckpt_pos.shape)} -> {tuple(state_dict[key].shape)}')


class TemporalAttentionFusion(nn.Module):
    def __init__(self, channels: int, num_levels: int):
        super().__init__()
        hidden = max(channels // 4, 64)
        self.score_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, 1))
            for _ in range(num_levels)
        ])

    def forward_one(self, x: torch.Tensor, level: int) -> torch.Tensor:
        # x: B x T x C x H x W
        b, t, c, h, w = x.shape
        xt = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
        score = self.score_mlps[level](xt)
        weight = torch.softmax(score, dim=1)
        fused = (xt * weight).sum(dim=1)
        return fused.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


@MODELS.register_module(force=True)
class CopernicusFMTemporalSegBackbone(BaseModule):
    """Official segmentation-version Copernicus-FM wrapper for PASTIS.

    Difference from the image-level model:
    this uses ``models_dwv_seg.py`` and returns dense intermediate feature maps
    from several transformer blocks, which is the correct interface for UPerNet
    or a linear segmentation probe.

    Input: B x T x C x H x W.
    Output: tuple of fused B x D x h x w feature maps.
    """

    def __init__(
        self,
        arch: str = 'base',
        pretrained: Optional[str] = None,
        out_indices: Optional[Sequence[int]] = None,
        img_size: int = 224,
        patch_size: int = 16,
        kernel_size: int = 16,
        wave_list: Optional[Sequence[float]] = None,
        bandwidth: Optional[Sequence[float]] = None,
        input_mode: str = 'spectral',
        var_option: str = 'spectrum',
        temporal_fusion: str = 'mean',
        time_chunk_size: Optional[int] = None,
        freeze_backbone: bool = False,
        meta_mode: str = 'nan',
        default_lon: float = float('nan'),
        default_lat: float = float('nan'),
        default_delta_time: float = float('nan'),
        default_patch_token_area: Optional[float] = None,
        pixel_size_m: float = 10.0,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        if arch not in ARCH_SETTINGS:
            raise KeyError(f'Unsupported arch={arch!r}. Available: {list(ARCH_SETTINGS)}')
        self.arch = arch
        self.embed_dim = ARCH_CHANNELS[arch]
        self.out_indices = tuple(out_indices) if out_indices is not None else ARCH_OUT_INDICES[arch]
        self.pretrained = pretrained
        self.kernel_size = kernel_size
        self.input_mode = input_mode
        self.var_option = var_option.lower()
        self.temporal_fusion = temporal_fusion.lower()
        self.time_chunk_size = time_chunk_size
        self.meta_mode = meta_mode.lower()
        self.default_lon = default_lon
        self.default_lat = default_lat
        self.default_delta_time = default_delta_time
        self.default_patch_token_area = default_patch_token_area
        self.pixel_size_m = pixel_size_m
        self._frozen = False

        self.wave_list = [float(x) for x in (wave_list or S2_13_WAVELENGTHS_NM)]
        self.bandwidth = [float(x) for x in (bandwidth or S2_13_BANDWIDTHS_NM)]
        if len(self.wave_list) != len(self.bandwidth):
            raise ValueError('wave_list and bandwidth must have the same length')

        arch_cfg = ARCH_SETTINGS[arch]
        # Build the official segmentation-version ViT directly. We avoid vit_* helper
        # functions here because they hard-code out_indices, while the OpenMMLab
        # config should be able to override it.
        self.cfm = CopernicusFMViT(
            img_size=img_size,
            patch_size=patch_size or arch_cfg['patch_size'],
            out_indices=list(self.out_indices),
            embed_dim=arch_cfg['embed_dim'],
            depth=arch_cfg['depth'],
            num_heads=arch_cfg['num_heads'],
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            # For Sentinel-2 spectral segmentation we do not need the variable-name
            # language embedding branch. Using 'spectrum' avoids any online download
            # of var_embed_llama3.2_1B.pt during model construction.
            var_option=self.var_option,
            loc_option='lonlat',
        )

        if self.temporal_fusion == 'attention':
            self.attention_fusion = TemporalAttentionFusion(self.embed_dim, len(self.out_indices))
        elif self.temporal_fusion in ('mean', 'avg', 'average'):
            self.attention_fusion = None
        else:
            raise ValueError(f'Unsupported temporal_fusion={temporal_fusion!r}. Use mean or attention.')

        if freeze_backbone:
            self.set_backbone_trainable(False)

    @property
    def out_channels(self):
        return [self.embed_dim for _ in self.out_indices]

    def init_weights(self):
        super().init_weights()
        if self.pretrained:
            state_dict = _load_checkpoint_state(self.pretrained)

            # 关键：在 load_state_dict 前，先把 checkpoint pos_embed 插值到当前 img_size
            _resize_pos_embed_for_current_model(state_dict, self.cfm, key='pos_embed')

            msg = self.cfm.load_state_dict(state_dict, strict=False)
            self._print_load_msg(msg)

        if self._frozen:
            self.set_backbone_trainable(False)

    @master_only
    def _print_load_msg(self, msg):
        missing = list(msg.missing_keys)
        unexpected = list(msg.unexpected_keys)
        print('[CopernicusFMTemporalSegBackbone] Loaded pretrained:', self.pretrained)
        print(f'[CopernicusFMTemporalSegBackbone] missing_keys={len(missing)}, unexpected_keys={len(unexpected)}')
        if missing:
            print('[CopernicusFMTemporalSegBackbone] first missing keys:', missing[:20])
        if unexpected:
            print('[CopernicusFMTemporalSegBackbone] first unexpected keys:', unexpected[:20])

    def set_backbone_trainable(self, trainable: bool = True):
        for p in self.parameters():
            p.requires_grad = trainable
        self._frozen = not trainable
        if not trainable:
            self.eval()

    def set_cfm_trainable(self, trainable: bool = True):
        for p in self.cfm.parameters():
            p.requires_grad = trainable
        if not trainable:
            self.cfm.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._frozen:
            super().train(False)
        return self

    def _build_meta(self, n: int, device, dtype):
        if self.meta_mode == 'nan':
            return torch.full((n, 4), float('nan'), device=device, dtype=dtype)
        if self.meta_mode != 'constant':
            raise ValueError(f'Unsupported meta_mode={self.meta_mode!r}. Use nan or constant.')
        area = self.default_patch_token_area
        if area is None:
            area = (self.kernel_size * self.pixel_size_m / 1000.0) ** 2
        meta = torch.tensor([self.default_lon, self.default_lat, self.default_delta_time, area], device=device, dtype=dtype)
        return meta.view(1, 4).repeat(n, 1)

    def _forward_cfm_chunks(self, x: torch.Tensor) -> List[torch.Tensor]:
        n = x.shape[0]
        chunk_size = self.time_chunk_size or n
        features_by_level = [[] for _ in self.out_indices]
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            x_chunk = x[start:end]
            meta = self._build_meta(end - start, x_chunk.device, x_chunk.dtype)
            feats = self.cfm(
                x_chunk,
                meta_info=meta,
                key=None,
                wave_list=self.wave_list,
                bandwidth=self.bandwidth,
                language_embed=None,
                input_mode=self.input_mode,
                kernel_size=self.kernel_size,
            )
            if len(feats) != len(self.out_indices):
                raise RuntimeError(f'Expected {len(self.out_indices)} features, got {len(feats)}. Check out_indices={self.out_indices}.')
            for i, feat in enumerate(feats):
                features_by_level[i].append(feat)
        return [torch.cat(chunks, dim=0) for chunks in features_by_level]

    def _fuse_one_level(self, feat: torch.Tensor, b: int, t: int, level: int) -> torch.Tensor:
        _, c, h, w = feat.shape
        feat = feat.reshape(b, t, c, h, w)
        if self.temporal_fusion in ('mean', 'avg', 'average'):
            return feat.mean(dim=1)
        if self.temporal_fusion == 'attention':
            return self.attention_fusion.forward_one(feat, level)
        raise AssertionError('unreachable')

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError(f'CopernicusFMTemporalSegBackbone expects B x T x C x H x W, got {tuple(x.shape)}')
        b, t, c, h, w = x.shape
        if c != len(self.wave_list):
            raise ValueError(f'Input channel C={c}, but len(wave_list)={len(self.wave_list)}. Check channel_map and wave_list.')
        x = x.reshape(b * t, c, h, w)
        flat_feats = self._forward_cfm_chunks(x)
        fused = [self._fuse_one_level(feat, b, t, i) for i, feat in enumerate(flat_feats)]
        return tuple(fused)
