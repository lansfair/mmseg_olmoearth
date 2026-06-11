from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from torch import Tensor


class _BandProjection(nn.Module):
    """Learnable per-timestep spectral adapter, matching the PASTIS script."""

    def __init__(self, in_channels: int, out_channels: int = 3) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class _MultiScaleFPN(nn.Module):
    """Fuse same-resolution ViT layer maps through a small synthetic pyramid."""

    def __init__(self, in_dim: int, out_dim: int, num_levels: int) -> None:
        super().__init__()
        self.projs = nn.ModuleList(
            [nn.Conv2d(in_dim, out_dim, kernel_size=1) for _ in range(num_levels)]
        )
        self.laterals = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_levels)
            ]
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim * num_levels, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, multi_feats: Sequence[Tensor]) -> Tensor:
        base_size = multi_feats[0].shape[-2:]
        scales = []
        for i, (feat, proj) in enumerate(zip(multi_feats, self.projs)):
            feat = proj(feat)
            if i == 0:
                feat = F.interpolate(
                    feat, scale_factor=2, mode="bilinear", align_corners=False
                )
            elif i == len(multi_feats) - 1:
                feat = F.interpolate(
                    feat, scale_factor=0.5, mode="bilinear", align_corners=False
                )
            scales.append(feat)

        for i in range(len(scales) - 1, -1, -1):
            if i < len(scales) - 1:
                up = F.interpolate(
                    scales[i + 1],
                    size=scales[i].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                scales[i] = scales[i] + up
            scales[i] = self.laterals[i](scales[i])

        fused = torch.cat(
            [
                F.interpolate(x, size=base_size, mode="bilinear", align_corners=False)
                for x in scales
            ],
            dim=1,
        )
        return self.fusion(fused)


class _RegLTAE(nn.Module):
    """Register-token-modulated temporal attention from the course example."""

    def __init__(self, dim: int, n_groups: int = 16) -> None:
        super().__init__()
        if dim % n_groups != 0:
            raise ValueError(f"dim={dim} must be divisible by n_groups={n_groups}")
        self.dim = dim
        self.n_groups = n_groups
        self.group_dim = dim // n_groups
        self.query = nn.Parameter(
            torch.randn(1, 1, 1, n_groups, self.group_dim) * 0.02
        )
        self.reg_mod = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, n_groups * self.group_dim),
        )
        self.key_proj = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_uniform_(self.key_proj.weight)
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: Tensor, registers: Tensor | None = None) -> Tensor:
        bsz, timesteps, num_tokens, dim = x.shape
        groups, group_dim = self.n_groups, self.group_dim
        keys = self.key_proj(x).view(bsz, timesteps, num_tokens, groups, group_dim)

        query = self.query
        if registers is not None:
            reg_ctx = registers.mean(dim=1, keepdim=True)
            reg_bias = self.reg_mod(reg_ctx).view(
                bsz, 1, 1, groups, group_dim
            )
            query = query + reg_bias

        scores = (keys * query).sum(dim=-1)
        attn = F.softmax(
            scores / (group_dim**0.5) * torch.exp(self.log_temp),
            dim=1,
        )
        values = x.view(bsz, timesteps, num_tokens, groups, group_dim)
        out = (attn.unsqueeze(-1) * values).sum(dim=1)
        return out.view(bsz, num_tokens, dim)


@MODELS.register_module()
class DINOv3PASTISTemporalBackbone(BaseModule):
    """PASTIS temporal DINOv3 backbone for MMSeg.

    Input is a flattened temporal Sentinel-2 tensor:
    ``(B, num_bands * num_timesteps, H, W)``.

    The module follows ``notebooks/01_pastis/train_pastis.py``:
    band projection -> frozen DINOv3 intermediates -> FPN -> RegLTAE.
    """

    def __init__(
        self,
        repo_dir: str,
        model_name: str = "dinov3_vitl16",
        weights_path: str | None = None,
        num_bands: int = 10,
        num_timesteps: int = 12,
        patch_size: int = 16,
        in_size: int | None = None,
        out_indices: Sequence[int] = (8, 16, 23),
        backbone_channels: int = 1024,
        out_channels: int = 256,
        tae_groups: int = 16,
        freeze_dinov3: bool = True,
        hub_kwargs: dict[str, Any] | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.repo_dir = str(repo_dir)
        self.model_name = model_name
        self.weights_path = str(weights_path) if weights_path else None
        self.num_bands = num_bands
        self.num_timesteps = num_timesteps
        self.patch_size = patch_size
        self.in_size = in_size
        self.out_indices = tuple(out_indices)
        self.backbone_channels = backbone_channels
        self.out_channels = out_channels
        self.freeze_dinov3 = freeze_dinov3
        self.hub_kwargs = hub_kwargs or {}

        self.band_proj = _BandProjection(num_bands, 3)
        self.model = self._load_model()
        self.fpn = _MultiScaleFPN(
            in_dim=backbone_channels,
            out_dim=out_channels,
            num_levels=len(self.out_indices),
        )
        self.reg2tae = nn.Linear(backbone_channels, out_channels)
        self.reg_tae = _RegLTAE(dim=out_channels, n_groups=tae_groups)

        if freeze_dinov3:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def _load_model(self):
        repo_dir = Path(self.repo_dir)
        if not repo_dir.exists():
            raise FileNotFoundError(f"DINOv3 repo_dir does not exist: {repo_dir}")
        kwargs = dict(self.hub_kwargs)
        if self.weights_path is not None:
            weights_path = Path(self.weights_path)
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"DINOv3 weights_path does not exist: {weights_path}"
                )
            kwargs.setdefault("weights", str(weights_path))
        return torch.hub.load(
            str(repo_dir),
            self.model_name,
            source="local",
            **kwargs,
        )

    def init_weights(self) -> None:
        return

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_dinov3:
            self.model.eval()
        return self

    def _extract_dinov3_features(self, x: Tensor) -> tuple[list[Tensor], Tensor]:
        with torch.set_grad_enabled(not self.freeze_dinov3):
            outputs = self.model.get_intermediate_layers(
                x,
                n=self.out_indices,
                reshape=True,
                norm=True,
                return_class_token=False,
                return_extra_tokens=True,
            )
        multi_feats = [item[0].contiguous() for item in outputs]
        registers = outputs[-1][1].contiguous()
        return multi_feats, registers

    def forward(self, inputs: Tensor) -> tuple[Tensor]:
        if inputs.ndim != 4:
            raise ValueError(
                "DINOv3PASTISTemporalBackbone expects BCHW input, "
                f"got shape {tuple(inputs.shape)}"
            )
        bsz, channels, height, width = inputs.shape
        expected = self.num_bands * self.num_timesteps
        if channels != expected:
            raise ValueError(
                f"Expected {expected} channels "
                f"({self.num_bands} bands x {self.num_timesteps} timesteps), "
                f"got {channels}"
            )
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                "Input size must be divisible by patch_size, "
                f"got {(height, width)} and patch_size={self.patch_size}"
            )

        x = inputs.view(
            bsz,
            self.num_bands,
            self.num_timesteps,
            height,
            width,
        )
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(bsz * self.num_timesteps, self.num_bands, height, width)
        x = self.band_proj(x)
        if self.in_size is not None and (height, width) != (self.in_size, self.in_size):
            x = F.interpolate(
                x,
                size=(self.in_size, self.in_size),
                mode="bilinear",
                align_corners=False,
            )

        multi_feats, registers = self._extract_dinov3_features(x)
        fused = self.fpn(multi_feats)
        feat_h, feat_w = fused.shape[-2:]
        num_tokens = feat_h * feat_w
        fused_seq = fused.flatten(2).transpose(1, 2)
        fused_seq = fused_seq.view(
            bsz,
            self.num_timesteps,
            num_tokens,
            self.out_channels,
        )

        registers = registers.view(bsz, self.num_timesteps, registers.shape[1], -1)
        registers = registers.mean(dim=1)
        registers = self.reg2tae(registers)
        agg = self.reg_tae(fused_seq, registers=registers)
        feat_map = agg.transpose(1, 2).view(
            bsz,
            self.out_channels,
            feat_h,
            feat_w,
        )
        return (feat_map.contiguous(),)
