from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from mmengine.runner.checkpoint import CheckpointLoader
from mmseg.registry import MODELS
from torch import Tensor, nn

from ..structures import EmbeddingResult, ModelCapabilities
from .base import BaseGeoFMAdapter


SPECTRAL_METADATA = {
    "sentinel2_l2a": {
        "bands": (
            "B02", "B03", "B04", "B08", "B05", "B06",
            "B07", "B8A", "B11", "B12", "B01", "B09",
        ),
        "wavelength": {
            "B01": 440, "B02": 490, "B03": 560, "B04": 665,
            "B05": 705, "B06": 740, "B07": 783, "B08": 842,
            "B8A": 860, "B09": 940, "B11": 1610, "B12": 2190,
        },
        "bandwidth": {
            "B01": 20, "B02": 65, "B03": 35, "B04": 30,
            "B05": 15, "B06": 15, "B07": 20, "B08": 115,
            "B8A": 20, "B09": 20, "B11": 90, "B12": 180,
        },
    },
    "sentinel1": {
        "bands": ("vv", "vh"),
        "wavelength": {"vv": 50000000, "vh": 50000000},
        "bandwidth": {"vv": 1e9, "vh": 1e9},
    },
}


def _clean_state_dict(checkpoint: Any) -> dict[str, Tensor]:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("CopernicusFM checkpoint must contain a state dict.")
    return {
        re.sub(r"^(module\.|model\.)+", "", key): value
        for key, value in checkpoint.items()
    }


@MODELS.register_module()
class CopernicusFMAdapter(BaseGeoFMAdapter):
    """Final-token CopernicusFM extraction matching the OlmoEarth wrapper."""

    model_family = "copernicusfm"

    def __init__(
        self,
        modalities: Sequence[str] = ("sentinel2_l2a",),
        model_variant: str = "base",
        temporal_pooling: str = "mean",
        image_size: int = 224,
        patch_size: int = 16,
        out_channels: int = 768,
        freeze: bool = True,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(model_variant=model_variant, init_cfg=init_cfg)
        if not modalities or not set(modalities).issubset(SPECTRAL_METADATA):
            raise ValueError("CopernicusFM supports sentinel1 and sentinel2_l2a.")
        if temporal_pooling not in {"mean", "max"}:
            raise ValueError("temporal_pooling must be 'mean' or 'max'.")
        self.modalities = tuple(modalities)
        self.temporal_pooling = temporal_pooling
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)
        self.freeze = freeze

        try:
            from projects.CopernicusBench.copernicusbench.models.copernicus_fm_backbone import (
                CopernicusFMBackbone,
            )
        except ImportError as exc:
            raise ImportError(
                "CopernicusFMAdapter requires projects.CopernicusBench."
            ) from exc
        self.backbone = CopernicusFMBackbone(
            arch=model_variant,
            input_mode="spectral",
            kernel_size=patch_size,
            frozen_exclude=[] if freeze else ["all"],
            init_cfg=None,
        )
        self.fc_norm = nn.LayerNorm(out_channels)
        if freeze:
            self.backbone.requires_grad_(False)
            self.fc_norm.requires_grad_(False)
            self.backbone.eval()
            self.fc_norm.eval()

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supported_modalities=frozenset(self.modalities),
            required_modalities=frozenset(self.modalities),
            supports_global=True,
            supports_dense=True,
            supports_multitemporal=True,
            supports_multimodal=True,
            native_stride=self.patch_size,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
            self.fc_norm.eval()
        return self

    def init_weights(self) -> None:
        if self.init_cfg is None:
            return
        if not isinstance(self.init_cfg, dict) or self.init_cfg.get("type") != "Pretrained":
            super().init_weights()
            return
        checkpoint_path = self.init_cfg.get("checkpoint")
        if checkpoint_path is None:
            raise ValueError("CopernicusFMAdapter requires checkpoint in init_cfg.")
        checkpoint = CheckpointLoader.load_checkpoint(
            checkpoint_path, map_location="cpu", logger=None
        )
        state_dict = _clean_state_dict(checkpoint)
        encoder_state = {
            key: value
            for key, value in state_dict.items()
            if key in self.backbone.encoder.state_dict()
        }
        incompatible = self.backbone.encoder.load_state_dict(
            encoder_state, strict=False
        )
        if incompatible.missing_keys:
            raise RuntimeError(
                "Missing CopernicusFM encoder keys: "
                + ", ".join(incompatible.missing_keys)
            )
        norm_state = {
            name: state_dict[f"fc_norm.{name}"]
            for name in ("weight", "bias")
            if f"fc_norm.{name}" in state_dict
        }
        if len(norm_state) != 2:
            raise RuntimeError("Checkpoint does not contain fc_norm weight and bias.")
        self.fc_norm.load_state_dict(norm_state, strict=True)
        if self.freeze:
            self.backbone.requires_grad_(False)
            self.fc_norm.requires_grad_(False)
        self._is_init = True

    def prepare_inputs(
        self,
        inputs: Any,
        batch_metainfo: Sequence[dict[str, Any]] | None = None,
    ) -> list[Tensor]:
        tensors = self.modality_tensors(inputs)
        timesteps = {tensors[name].shape[1] for name in self.modalities}
        if len(timesteps) != 1:
            raise ValueError("CopernicusFM modalities must share timestep count.")
        timestep_count = next(iter(timesteps))
        wavelengths = []
        bandwidths = []
        for name in self.modalities:
            value = tensors[name]
            if value.ndim != 5:
                raise ValueError(f"{name} must have shape [B,T,C,H,W].")
            bands = SPECTRAL_METADATA[name]["bands"]
            if value.shape[2] != len(bands):
                raise ValueError(
                    f"{name} expected {len(bands)} bands, got {value.shape[2]}."
                )
            wavelengths.extend(SPECTRAL_METADATA[name]["wavelength"][b] for b in bands)
            bandwidths.extend(SPECTRAL_METADATA[name]["bandwidth"][b] for b in bands)
        self.backbone.band_wavelengths = wavelengths
        self.backbone.band_bandwidths = bandwidths

        output = []
        for timestep in range(timestep_count):
            image = torch.cat(
                [tensors[name][:, timestep] for name in self.modalities], dim=1
            )
            output.append(
                F.interpolate(
                    image,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return output

    def _pool_time(self, features: list[Tensor]) -> Tensor:
        stacked = torch.stack(features, dim=1)
        if self.temporal_pooling == "mean":
            return stacked.mean(dim=1)
        return stacked.max(dim=1).values

    def extract_dense(self, prepared_inputs: list[Tensor]) -> Tensor:
        return self._pool_time([self.backbone(image)[-1] for image in prepared_inputs])

    def extract_global(self, prepared_inputs: list[Tensor]) -> Tensor:
        features = [
            self.fc_norm(self.backbone(image)[-1].mean(dim=(-2, -1)))
            for image in prepared_inputs
        ]
        return self._pool_time(features)

    def extract(self, inputs, batch_metainfo=None, mode="dense") -> EmbeddingResult:
        result = super().extract(inputs, batch_metainfo, mode)
        result.pooling = self.temporal_pooling
        result.metadata.update(
            {
                "image_size": self.image_size,
                "global_fc_norm": mode == "global",
            }
        )
        return result
