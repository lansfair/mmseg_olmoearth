from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from mmseg.registry import MODELS
from torch import Tensor

from ..structures import EmbeddingResult, ModelCapabilities
from .base import BaseGeoFMAdapter


RGB_INDICES = {
    # OlmoEarth modality band order: B02, B03, B04, ...
    "sentinel2_l2a": (2, 1, 0),
    # OlmoEarth modality band order: B8, B1, B2, B3, B4, ...
    "landsat": (4, 3, 2),
}


@MODELS.register_module()
class DINOv3Adapter(BaseGeoFMAdapter):
    """OlmoEarth-style DINOv3 RGB patch-token extraction."""

    model_family = "dinov3"

    def __init__(
        self,
        repo_dir: str,
        modality: str = "sentinel2_l2a",
        model_name: str = "dinov3_vitl16",
        model_variant: str = "vitl16-sat493m",
        weights_path: str | None = None,
        patch_size: int = 16,
        out_channels: int = 1024,
        temporal_pooling: str = "mean",
        base_resize: int = 256,
        apply_normalization: bool = False,
        satellite_normalization: bool = True,
        freeze: bool = True,
        hub_kwargs: dict[str, Any] | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(model_variant=model_variant, init_cfg=init_cfg)
        if modality not in RGB_INDICES:
            raise ValueError(
                "DINOv3Adapter supports sentinel2_l2a or landsat, "
                f"got {modality!r}."
            )
        if temporal_pooling not in {"mean", "max"}:
            raise ValueError("temporal_pooling must be 'mean' or 'max'.")
        self.modality = modality
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)
        self.temporal_pooling = temporal_pooling
        self.base_resize = int(base_resize)
        self.apply_normalization = apply_normalization

        try:
            from projects.dinov3.dinov3.backbones.dinov3_backbone import (
                DINOv3ViTBackbone,
            )
        except ImportError as exc:
            raise ImportError(
                "DINOv3Adapter requires projects.dinov3.dinov3."
            ) from exc
        self.backbone = DINOv3ViTBackbone(
            repo_dir=repo_dir,
            model_name=model_name,
            weights_path=weights_path,
            patch_size=patch_size,
            out_channels=out_channels,
            freeze=freeze,
            out_indices=None,
            hub_kwargs=hub_kwargs,
        )
        if satellite_normalization:
            mean = (0.430, 0.411, 0.296)
            std = (0.213, 0.156, 0.143)
        else:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        self.register_buffer("normalization_mean", torch.tensor(mean)[None, :, None, None])
        self.register_buffer("normalization_std", torch.tensor(std)[None, :, None, None])

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supported_modalities=frozenset({self.modality}),
            required_modalities=frozenset({self.modality}),
            supports_global=True,
            supports_dense=True,
            supports_multitemporal=True,
            supports_multimodal=False,
            native_stride=self.patch_size,
        )

    @staticmethod
    def select_rgb(value: Tensor, modality: str) -> Tensor:
        if value.ndim != 5:
            raise ValueError(
                "DINOv3 canonical input must have shape [B,T,C,H,W]."
            )
        if value.shape[2] == 3:
            return value
        indices = RGB_INDICES[modality]
        if value.shape[2] <= max(indices):
            raise ValueError(
                f"{modality} has only {value.shape[2]} channels; cannot select RGB."
            )
        return value[:, :, indices, :, :]

    def prepare_inputs(
        self,
        inputs: Any,
        batch_metainfo: Sequence[dict[str, Any]] | None = None,
    ) -> list[Tensor]:
        value = self.modality_tensors(inputs)[self.modality]
        value = self.select_rgb(value, self.modality)
        per_timestep = []
        for index in range(value.shape[1]):
            image = value[:, index]
            height = image.shape[-2]
            resize = height if height > self.base_resize else self.base_resize
            image = F.interpolate(
                image,
                size=(resize, resize),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            if self.apply_normalization:
                image = (image - self.normalization_mean) / self.normalization_std
            per_timestep.append(image)
        return per_timestep

    def _pool_time(self, features: list[Tensor]) -> Tensor:
        stacked = torch.stack(features, dim=1)
        if self.temporal_pooling == "mean":
            return stacked.mean(dim=1)
        return stacked.max(dim=1).values

    def extract_dense(self, prepared_inputs: list[Tensor]) -> Tensor:
        features = [self.backbone(image)[-1] for image in prepared_inputs]
        return self._pool_time(features)

    def extract_global(self, prepared_inputs: list[Tensor]) -> Tensor:
        features = [
            self.backbone(image)[-1].mean(dim=(-2, -1))
            for image in prepared_inputs
        ]
        return self._pool_time(features)

    def extract(self, inputs, batch_metainfo=None, mode="dense") -> EmbeddingResult:
        result = super().extract(inputs, batch_metainfo, mode)
        result.pooling = self.temporal_pooling
        result.metadata.update(
            {
                "rgb_only": True,
                "apply_normalization": self.apply_normalization,
                "base_resize": self.base_resize,
            }
        )
        return result
