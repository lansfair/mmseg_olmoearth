from __future__ import annotations

import re
from typing import Any

import torch
from mmengine.logging import MMLogger
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import CheckpointLoader
from mmseg.registry import MODELS
from torch import Tensor

from .swin_transformer_source import swin_huge


@MODELS.register_module()
class DINOv3DistilledSwinHuge(BaseModule):
    """DINOv3-distilled Swin-Huge backbone for MMSegmentation.

    The distillation project trains a DINOv3-style Swin student whose normal
    forward returns token dictionaries. This wrapper uses the same Swin-Huge
    source and weights, but returns four feature maps at strides 4/8/16/32.
    """

    out_channels = (352, 704, 1408, 2816)

    def __init__(
        self,
        checkpoint: str | None = None,
        img_size: int = 224,
        patch_size: int = 4,
        window_size: int = 8,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
        checkpoint_prefix: str = "auto",
        use_ema: bool = True,
        strict: bool = False,
        final_norm: bool = True,
        frozen: bool = False,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.checkpoint = checkpoint
        self.out_indices = tuple(out_indices)
        self.checkpoint_prefix = checkpoint_prefix
        self.use_ema = use_ema
        self.strict = strict
        self.final_norm = final_norm
        self.frozen = frozen
        self.out_channels = tuple(self.out_channels[i] for i in self.out_indices)

        self.model = swin_huge(
            img_size=img_size,
            patch_size=patch_size,
            window_size=window_size,
            qkv_bias=True,
            norm_layer="layernorm",
            n_storage_tokens=0,
            untie_cls_and_patch_norms=True,
            untie_global_and_local_cls_norm=False,
        )

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "teacher"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    return checkpoint[key]
        if not isinstance(checkpoint, dict):
            raise TypeError("Checkpoint must be a state_dict or contain one.")
        return checkpoint

    def _candidate_prefixes(self) -> tuple[str, ...]:
        if self.checkpoint_prefix != "auto":
            return (self.checkpoint_prefix,)
        if self.use_ema:
            return (
                "model.model_ema.backbone.",
                "model_ema.backbone.",
                "model.student.backbone.",
                "student.backbone.",
                "backbone.",
            )
        return (
            "model.student.backbone.",
            "student.backbone.",
            "model.model_ema.backbone.",
            "model_ema.backbone.",
            "backbone.",
        )

    def _clean_state_dict(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        model_state = self.model.state_dict()
        best: dict[str, Tensor] = {}

        for prefix in self._candidate_prefixes():
            cleaned = {}
            for key, value in state_dict.items():
                clean_key = re.sub(r"^(module\.)+", "", key)
                if not clean_key.startswith(prefix):
                    continue
                clean_key = clean_key[len(prefix):]
                if (
                    clean_key in model_state
                    and tuple(model_state[clean_key].shape) == tuple(value.shape)
                ):
                    cleaned[clean_key] = value
            if len(cleaned) > len(best):
                best = cleaned

        if best:
            return best

        cleaned = {}
        for key, value in state_dict.items():
            clean_key = re.sub(r"^(module\.)+", "", key)
            if (
                clean_key in model_state
                and tuple(model_state[clean_key].shape) == tuple(value.shape)
            ):
                cleaned[clean_key] = value
        return cleaned

    def init_weights(self) -> None:
        if self.checkpoint is None:
            return
        checkpoint = CheckpointLoader.load_checkpoint(
            self.checkpoint,
            map_location="cpu",
            logger=None,
        )
        state_dict = self._clean_state_dict(self._extract_state_dict(checkpoint))
        if not state_dict:
            raise RuntimeError(
                "No Swin-Huge backbone weights matched the checkpoint. "
                "Use an extracted backbone .pt or set checkpoint_prefix."
            )
        msg = self.model.load_state_dict(state_dict, strict=self.strict)
        logger = MMLogger.get_current_instance()
        logger.info(
            "Loaded %d distilled Swin-Huge tensors from %s; missing=%d unexpected=%d",
            len(state_dict),
            self.checkpoint,
            len(msg.missing_keys),
            len(msg.unexpected_keys),
        )
        self._is_init = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    @staticmethod
    def _tokens_to_map(tokens: Tensor, height: int, width: int) -> Tensor:
        batch, length, channels = tokens.shape
        if length != height * width:
            raise ValueError(
                f"Cannot reshape {length} tokens to feature map {(height, width)}."
            )
        return tokens.transpose(1, 2).reshape(batch, channels, height, width).contiguous()

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        _, _, input_h, input_w = x.shape
        patch = self.model.patch_embed.patch_size[0]
        if input_h % patch != 0 or input_w % patch != 0:
            raise ValueError(
                f"Input size {(input_h, input_w)} must be divisible by patch_size={patch}."
            )

        height, width = input_h // patch, input_w // patch
        if height != width:
            raise ValueError(
                "The distilled Swin source assumes square token grids; "
                f"got patch grid {(height, width)}."
            )

        tokens = self.model.patch_embed(x).flatten(2).transpose(1, 2)
        if self.model.ape:
            tokens = tokens + self.model.absolute_pos_embed
        tokens = self.model.pos_drop(tokens)

        outs = []
        for stage_idx, layer in enumerate(self.model.layers):
            for block in layer.blocks:
                tokens, _ = block(tokens)

            stage_tokens = tokens
            if stage_idx == len(self.model.layers) - 1 and self.final_norm:
                stage_tokens = self.model.norm(stage_tokens)
            if stage_idx in self.out_indices:
                outs.append(self._tokens_to_map(stage_tokens, height, width))

            if layer.downsample is not None:
                tokens = layer.downsample(tokens)
                height //= 2
                width //= 2

        return tuple(outs)
