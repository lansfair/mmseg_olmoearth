from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from mmengine.model import BaseModule
from mmseg.registry import MODELS
from torch import Tensor

from .dinov3_adapter_source import DINOv3_Adapter


def _default_repo_dir() -> str:
    return str(Path(__file__).resolve().parents[2] / "dinov3-main")


def _add_repo_to_path(repo_dir: str) -> None:
    repo_path = Path(repo_dir)
    if not repo_path.exists():
        raise FileNotFoundError(f"DINOv3 repo_dir does not exist: {repo_path}")
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))


@MODELS.register_module()
class DINOv3AdapterBackbone(BaseModule):
    """DINOv3 ViT + official segmentation adapter for MMSegmentation.

    Returns four feature maps at strides 4/8/16/32. For ViT-L/16 each feature
    map has 1024 channels.
    """

    default_interaction_indexes = {
        "vit_small": [2, 5, 8, 11],
        "vit_base": [2, 5, 8, 11],
        "vit_large": [5, 11, 17, 23],
        "vit_huge": [7, 15, 23, 31],
    }

    vitl16_cfg = dict(
        pos_embed_rope_base=100.0,
        pos_embed_rope_min_period=None,
        pos_embed_rope_max_period=None,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        norm_layer="layernormbf16",
        n_storage_tokens=4,
        mask_k_bias=True,
    )

    def __init__(
        self,
        repo_dir: str | None = None,
        arch: str = "vit_large",
        patch_size: int = 16,
        weights_path: str | None = None,
        weight_variant: str = "sat493m",
        interaction_indexes: list[int] | None = None,
        freeze_vit: bool = True,
        finetune_vit: bool = False,
        replace_ms_deform_attn: bool = True,
        with_cp: bool = False,
        vit_cfg_overrides: dict[str, Any] | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=None)
        self.repo_dir = repo_dir or _default_repo_dir()
        self.arch = arch
        self.patch_size = patch_size
        self.weights_path = weights_path
        self.freeze_vit = freeze_vit
        self.finetune_vit = finetune_vit
        if not replace_ms_deform_attn:
            raise ValueError(
                "DINOv3AdapterBackbone uses the local DINOv3 adapter with "
                "MMCV MSDeformAttn; replace_ms_deform_attn must be True."
            )
        self.out_channels = self._infer_out_channels(arch)

        _add_repo_to_path(self.repo_dir)

        from omegaconf import OmegaConf
        from dinov3.models import build_model_for_eval

        cfg = OmegaConf.create(
            {
                "student": {
                    "arch": arch,
                    "patch_size": patch_size,
                    "pos_embed_rope_base": None,
                    "pos_embed_rope_min_period": 4,
                    "pos_embed_rope_max_period": 50,
                    "pos_embed_rope_normalize_coords": "separate",
                    "pos_embed_rope_shift_coords": None,
                    "pos_embed_rope_jitter_coords": None,
                    "pos_embed_rope_rescale_coords": None,
                    "qkv_bias": True,
                    "layerscale": 1e-5,
                    "norm_layer": "layernorm",
                    "ffn_layer": "mlp",
                    "ffn_bias": True,
                    "proj_bias": True,
                    "n_storage_tokens": 0,
                    "mask_k_bias": False,
                    "untie_cls_and_patch_norms": False,
                    "untie_global_and_local_cls_norm": False,
                    "fp8_enabled": False,
                },
                "crops": {"global_crops_size": 224},
            }
        )

        for key, value in self._default_vit_overrides(weight_variant).items():
            cfg.student[key] = value
        if vit_cfg_overrides:
            for key, value in vit_cfg_overrides.items():
                cfg.student[key] = value

        vit = build_model_for_eval(cfg, pretrained_weights=weights_path)
        if interaction_indexes is None:
            interaction_indexes = self.default_interaction_indexes.get(
                arch, [2, 5, 8, 11]
            )

        self.adapter = DINOv3_Adapter(
            vit,
            interaction_indexes=interaction_indexes,
            with_cp=with_cp,
        )

        self.adapter.finetune_vit = finetune_vit
        self._set_trainable_parameters()

    @classmethod
    def _default_vit_overrides(cls, weight_variant: str) -> dict[str, Any]:
        overrides = {}
        variant = weight_variant.lower()
        if variant in {"lvd1689m", "sat493m"}:
            overrides.update(cls.vitl16_cfg)
        if variant == "sat493m":
            overrides["untie_global_and_local_cls_norm"] = True
        elif variant not in {"lvd1689m", "sat493m", "custom", "none"}:
            raise ValueError(
                "weight_variant must be one of 'lvd1689m', 'sat493m', "
                f"'custom', or 'none', got {weight_variant!r}."
            )
        return overrides

    @staticmethod
    def _infer_out_channels(arch: str) -> tuple[int, int, int, int]:
        channels = {
            "vit_small": 384,
            "vit_base": 768,
            "vit_large": 1024,
            "vit_huge": 1280,
        }.get(arch)
        if channels is None:
            raise ValueError(f"Unsupported DINOv3 arch: {arch}")
        return (channels, channels, channels, channels)

    def _set_trainable_parameters(self) -> None:
        self.adapter.requires_grad_(True)
        if self.freeze_vit and not self.finetune_vit:
            self.adapter.backbone.requires_grad_(False)
        else:
            self.adapter.backbone.requires_grad_(True)

    def init_weights(self) -> None:
        return

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vit and not self.finetune_vit:
            self.adapter.backbone.eval()
        return self

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        out = self.adapter(x)
        return (out["1"], out["2"], out["3"], out["4"])
